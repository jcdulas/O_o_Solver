#include "oos/reduction/cache/CacheFile.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/FileError.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/platform/TemporaryFile.hpp"
#include "oos/reduction/cache/CacheHasher.hpp"

#include <array>
#include <bit>
#include <fstream>
#include <limits>
#include <string>
#include <system_error>
#include <utility>

namespace oos::reduction
{

namespace
{

constexpr std::size_t WORD_BYTES = 8;
constexpr std::size_t RIGID_BODY_DOFS = 6;
constexpr std::size_t INERTIAL_PATTERN_COUNT = 6;
// A text of the header longer than this is not a text of O_o_Solver (a version, a kernel name, a model tag, a PARAM):
// the file is refused before anything is allocated for it.
constexpr std::uint64_t LONGEST_TEXT = 4096;
constexpr std::uint64_t MOST_PARAMETERS = 64;
// The largest count of rows or columns of a section: 2^26 boundary DOFs would make a K* of 2^51 values (16 PB), far
// beyond any unit (about 37 500 boundary DOFs for the largest); below it every size of a file fits 64
// bits, so that a file with other counts is refused as not O_o_Solver's before anything is allocated.
constexpr std::uint64_t MOST_SECTION_ROWS = std::uint64_t{1} << 26U;

// "NSVCACHE" as a little-endian word.
constexpr std::uint64_t magicWord()
{
    constexpr std::string_view MAGIC = "NSVCACHE";
    std::uint64_t word = 0;
    for (std::size_t b = 0; b < WORD_BYTES; ++b)
        word |= static_cast<std::uint64_t>(static_cast<unsigned char>(MAGIC[b])) << (8 * b);
    return word;
}
constexpr std::uint64_t MAGIC_WORD = magicWord();
// Read as this value only when the reader decodes the words as they were written (little-endian).
constexpr std::uint64_t BYTE_ORDER_MARK = 0x0102030405060708ULL;

// The words of an entry written to a stream, little-endian, each one mixed into the checksum.
class EntryWriter
{
public:
    explicit EntryWriter(std::ostream& out) : m_out(&out) {}

    void word(std::uint64_t value)
    {
        std::array<char, WORD_BYTES> bytes{};
        for (std::size_t b = 0; b < WORD_BYTES; ++b)
            bytes[b] = static_cast<char>((value >> (8 * b)) & 0xFFU);
        m_out->write(bytes.data(), static_cast<std::streamsize>(WORD_BYTES));
        m_hasher.mix(value);
        m_bytes += WORD_BYTES;
    }
    void signedWord(std::int64_t value) { word(std::bit_cast<std::uint64_t>(value)); }
    void real(double value) { word(std::bit_cast<std::uint64_t>(value)); }
    void text(std::string_view value)
    {
        word(static_cast<std::uint64_t>(value.size()));
        for (std::size_t first = 0; first < value.size(); first += WORD_BYTES)
        {
            std::uint64_t packed = 0;
            for (std::size_t b = 0; b < WORD_BYTES && first + b < value.size(); ++b)
                packed |= static_cast<std::uint64_t>(static_cast<unsigned char>(value[first + b])) << (8 * b);
            word(packed);
        }
    }
    void reals(const double* values, std::size_t count)
    {
        if constexpr (std::endian::native == std::endian::little)
        {
            // The words as they lie in memory: one write per column instead of one per value.
            m_out->write(reinterpret_cast<const char*>(values), static_cast<std::streamsize>(count * WORD_BYTES));
            for (std::size_t k = 0; k < count; ++k)
                m_hasher.mix(values[k]);
            m_bytes += count * WORD_BYTES;
        }
        else
        {
            for (std::size_t k = 0; k < count; ++k)
                real(values[k]);
        }
    }
    // The checksum of everything written, written last (not mixed into itself).
    void finish()
    {
        const std::uint64_t checksum = m_hasher.value();
        std::array<char, WORD_BYTES> bytes{};
        for (std::size_t b = 0; b < WORD_BYTES; ++b)
            bytes[b] = static_cast<char>((checksum >> (8 * b)) & 0xFFU);
        m_out->write(bytes.data(), static_cast<std::streamsize>(WORD_BYTES));
        m_bytes += WORD_BYTES;
    }
    [[nodiscard]] std::uint64_t bytes() const noexcept { return m_bytes; }

private:
    std::ostream* m_out;
    CacheHasher m_hasher;
    std::uint64_t m_bytes = 0;
};

// The words of an entry read from a stream, each one mixed into the checksum; a short read marks it truncated.
class EntryReader
{
public:
    explicit EntryReader(std::istream& in) : m_in(&in) {}

    bool word(std::uint64_t& value)
    {
        std::array<unsigned char, WORD_BYTES> bytes{};
        if (!m_in->read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(WORD_BYTES)))
        {
            m_isTruncated = true;
            return false;
        }
        value = 0;
        for (std::size_t b = 0; b < WORD_BYTES; ++b)
            value |= static_cast<std::uint64_t>(bytes[b]) << (8 * b);
        m_hasher.mix(value);
        m_bytes += WORD_BYTES;
        return true;
    }
    bool signedWord(std::int64_t& value)
    {
        std::uint64_t bits = 0;
        const bool isRead = word(bits);
        value = std::bit_cast<std::int64_t>(bits);
        return isRead;
    }
    bool real(double& value)
    {
        std::uint64_t bits = 0;
        const bool isRead = word(bits);
        value = std::bit_cast<double>(bits);
        return isRead;
    }
    bool text(std::string& value)
    {
        std::uint64_t length = 0;
        if (!word(length))
            return false;
        if (length > LONGEST_TEXT)
        {
            m_isMalformed = true;
            return false;
        }
        value.assign(static_cast<std::size_t>(length), '\0');
        for (std::size_t first = 0; first < value.size(); first += WORD_BYTES)
        {
            std::uint64_t packed = 0;
            if (!word(packed))
                return false;
            for (std::size_t b = 0; b < WORD_BYTES && first + b < value.size(); ++b)
                value[first + b] = static_cast<char>((packed >> (8 * b)) & 0xFFU);
        }
        return true;
    }
    bool reals(double* values, std::size_t count)
    {
        if constexpr (std::endian::native == std::endian::little)
        {
            if (!m_in->read(reinterpret_cast<char*>(values), static_cast<std::streamsize>(count * WORD_BYTES)))
            {
                m_isTruncated = true;
                return false;
            }
            for (std::size_t k = 0; k < count; ++k)
                m_hasher.mix(values[k]);
            m_bytes += count * WORD_BYTES;
            return true;
        }
        else
        {
            for (std::size_t k = 0; k < count; ++k)
                if (!real(values[k]))
                    return false;
            return true;
        }
    }
    // Whether the trailing checksum equals the checksum of every word read.
    bool isChecksumRight()
    {
        const std::uint64_t computed = m_hasher.value();
        std::array<unsigned char, WORD_BYTES> bytes{};
        if (!m_in->read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(WORD_BYTES)))
        {
            m_isTruncated = true;
            return false;
        }
        m_bytes += WORD_BYTES;
        std::uint64_t stored = 0;
        for (std::size_t b = 0; b < WORD_BYTES; ++b)
            stored |= static_cast<std::uint64_t>(bytes[b]) << (8 * b);
        return stored == computed;
    }
    [[nodiscard]] bool isTruncated() const noexcept { return m_isTruncated; }
    [[nodiscard]] bool isMalformed() const noexcept { return m_isMalformed; }
    void markMalformed() noexcept { m_isMalformed = true; }
    [[nodiscard]] std::uint64_t bytes() const noexcept { return m_bytes; }

private:
    std::istream* m_in;
    CacheHasher m_hasher;
    std::uint64_t m_bytes = 0;
    bool m_isTruncated = false;
    bool m_isMalformed = false;
};

std::uint64_t labelWord(const UnitDofLabel& label)
{
    return static_cast<std::uint64_t>(label.kind) | (static_cast<std::uint64_t>(label.side) << 8U) |
           (static_cast<std::uint64_t>(label.component) << 16U) |
           (static_cast<std::uint64_t>(static_cast<std::uint32_t>(label.id)) << 32U);
}

UnitDofLabel labelOfWord(std::uint64_t word)
{
    UnitDofLabel label;
    label.kind = static_cast<UnitPointKind>(word & 0xFFU);
    label.side = static_cast<FastenerSide>((word >> 8U) & 0xFFU);
    label.component = static_cast<std::uint8_t>((word >> 16U) & 0xFFU);
    label.id = static_cast<base::EntityId>(static_cast<std::int32_t>(static_cast<std::uint32_t>(word >> 32U)));
    return label;
}

void writeHeader(EntryWriter& writer, const CacheFileHeader& header)
{
    writer.word(MAGIC_WORD);
    writer.word(CacheFile::FORMAT_VERSION);
    writer.word(BYTE_ORDER_MARK);
    writer.word(static_cast<std::uint64_t>(header.kind));
    writer.signedWord(header.createdAt);
    writer.text(header.modelTag);
    const CacheKey& key = header.key;
    writer.text(key.solverVersion);
    writer.text(key.kernel);
    writer.word(key.elementFormulation);
    writer.signedWord(key.unitId);
    writer.word(key.unitIdentity);
    writer.signedWord(key.revision);
    writer.word(key.designStamp);
    writer.word(key.constraintStamp);
    writer.word(key.boundaryStamp);
    writer.word(key.spiderStamp);
    writer.word(static_cast<std::uint64_t>(key.reduction));
    writer.signedWord(key.modeCount);
    writer.word(key.modeCutoff);
    writer.word(static_cast<std::uint64_t>(key.parameters.size()));
    for (const CacheParameter& parameter : key.parameters)
    {
        writer.text(parameter.name);
        writer.text(parameter.value);
    }
    writer.word(key.patternStamp);
    writer.word(key.inputStamp);
    writer.word(key.equationStamp);
}

// Reads the header; on a refusal returns its text (empty: read).
std::string readHeaderWords(EntryReader& reader, CacheFileHeader& header)
{
    std::uint64_t magic = 0;
    if (!reader.word(magic) || magic != MAGIC_WORD)
        return "not a O_o_Solver cache file";
    std::uint64_t version = 0;
    if (!reader.word(version))
        return "truncated";
    if (version != CacheFile::FORMAT_VERSION)
        return base::concatenate({"cache file format ", std::to_string(version), ", this version of O_o_Solver reads ",
                                  std::to_string(CacheFile::FORMAT_VERSION)});
    std::uint64_t mark = 0;
    if (!reader.word(mark))
        return "truncated";
    if (mark != BYTE_ORDER_MARK)
        return "not a O_o_Solver cache file (its byte order mark is wrong)";
    std::uint64_t kind = 0;
    if (!reader.word(kind))
        return "truncated";
    if (kind != static_cast<std::uint64_t>(CacheEntryKind::Reduction) &&
        kind != static_cast<std::uint64_t>(CacheEntryKind::RecoveryData))
        return base::concatenate({"an entry of unknown kind ", std::to_string(kind)});
    header.kind = static_cast<CacheEntryKind>(kind);
    CacheKey& key = header.key;
    std::uint64_t reduction = 0;
    std::int64_t modeCount = 0;
    std::uint64_t parameterCount = 0;
    std::uint64_t elementFormulation = 0;
    const bool isRead = reader.signedWord(header.createdAt) && reader.text(header.modelTag) &&
                        reader.text(key.solverVersion) && reader.text(key.kernel) && reader.word(elementFormulation) &&
                        reader.signedWord(key.unitId) && reader.word(key.unitIdentity) &&
                        reader.signedWord(key.revision) && reader.word(key.designStamp) &&
                        reader.word(key.constraintStamp) && reader.word(key.boundaryStamp) &&
                        reader.word(key.spiderStamp) && reader.word(reduction) && reader.signedWord(modeCount) &&
                        reader.word(key.modeCutoff) && reader.word(parameterCount);
    if (!isRead)
        return reader.isMalformed() ? "corrupted (a text of its header is too long)" : "truncated";
    if (reduction > static_cast<std::uint64_t>(ReductionKind::CraigBampton) || parameterCount > MOST_PARAMETERS ||
        elementFormulation > std::numeric_limits<std::uint32_t>::max())
        return "corrupted (its header is not one O_o_Solver writes)";
    key.elementFormulation = static_cast<std::uint32_t>(elementFormulation);
    key.reduction = static_cast<ReductionKind>(reduction);
    key.modeCount = static_cast<std::int32_t>(modeCount);
    key.parameters.resize(static_cast<std::size_t>(parameterCount));
    for (CacheParameter& parameter : key.parameters)
        if (!reader.text(parameter.name) || !reader.text(parameter.value))
            return reader.isMalformed() ? "corrupted (a text of its header is too long)" : "truncated";
    if (!reader.word(key.patternStamp) || !reader.word(key.inputStamp) || !reader.word(key.equationStamp))
        return "truncated";
    return {};
}

std::uintmax_t fileSizeOf(const std::filesystem::path& path)
{
    std::error_code error;
    const std::uintmax_t size = std::filesystem::file_size(path, error);
    return error ? 0 : size;
}

// The words of a lower triangle of an n × n matrix, n(n + 1)/2, refused when it does not fit 64 bits.
bool triangleWords(std::uint64_t n, std::uint64_t& words)
{
    if (n > 0 && n > (std::numeric_limits<std::uint64_t>::max() / WORD_BYTES) / (n + 1))
        return false;
    words = n * (n + 1) / 2;
    return true;
}

// The total size of a file whose payload has `payloadWords` words after `headerBytes` (the checksum included); false
// when it does not fit 64 bits.
bool expectedSize(std::uint64_t headerBytes, std::uint64_t payloadWords, std::uint64_t& bytes)
{
    if (payloadWords > (std::numeric_limits<std::uint64_t>::max() - headerBytes) / WORD_BYTES - 1)
        return false;
    bytes = headerBytes + (payloadWords + 1) * WORD_BYTES;
    return true;
}

template <typename T>
CacheFileRead<T> refused(std::string reason, std::uint64_t bytesRead = 0)
{
    CacheFileRead<T> read;
    read.refusal = std::move(reason);
    read.bytesRead = bytesRead;
    return read;
}

std::uint64_t textWords(std::string_view text)
{
    return 1 + (text.size() + WORD_BYTES - 1) / WORD_BYTES;
}

// The words writeHeader writes.
std::uint64_t headerWords(const CacheFileHeader& header)
{
    constexpr std::uint64_t FIXED_WORDS = 5; // magic, version, byte order mark, kind, creation time
    constexpr std::uint64_t KEY_WORDS = 11;  // element formulation, unit ID, identity, revision, design, constraints,
                                             // boundary, spiders, reduction, modes, mode cut-off
    constexpr std::uint64_t STAMP_WORDS = 3; // patterns, inputs, equations
    std::uint64_t words = FIXED_WORDS + textWords(header.modelTag) + textWords(header.key.solverVersion) +
                          textWords(header.key.kernel) + KEY_WORDS + 1 + STAMP_WORDS;
    for (const CacheParameter& parameter : header.key.parameters)
        words += textWords(parameter.name) + textWords(parameter.value);
    return words;
}

std::uint64_t writeEntry(const std::filesystem::path& path, const CacheFileHeader& header,
                         const std::function<void(EntryWriter&)>& writePayload)
{
    base::TemporaryFile file(path);
    EntryWriter writer(file.stream());
    writeHeader(writer, header);
    writePayload(writer);
    writer.finish();
    file.commit();
    return writer.bytes();
}

} // namespace

namespace
{

// The parts of a reduction entry (the first word of its payload).
constexpr std::uint64_t HAS_UNIT_PART = 1;
constexpr std::uint64_t HAS_BOUNDARY_MASS = 2;
constexpr std::uint64_t HAS_MODAL_PART = 4;
constexpr std::uint64_t KNOWN_PARTS = HAS_UNIT_PART | HAS_BOUNDARY_MASS | HAS_MODAL_PART;
// The counts after the parts word: boundary DOFs, patterns, work, interior DOFs, pivot ratio, pattern stamp, unit ID,
// the modal part's boundary DOFs and modes.
constexpr std::uint64_t REDUCTION_COUNT_WORDS = 9;

// The counts of a reduction entry's payload.
struct ReductionCounts
{
    std::uint64_t parts = 0;
    std::uint64_t n = 0;        // the unit part's boundary DOFs (and the boundary mass's)
    std::uint64_t patterns = 0; // f*_k
    std::uint64_t work = 0;     // the pattern work's order
    std::uint64_t modalBoundary = 0;
    std::uint64_t modes = 0;
};

ReductionCounts countsOf(const CachedReduction& entry)
{
    ReductionCounts counts;
    if (entry.reduced)
    {
        counts.parts |= HAS_UNIT_PART;
        counts.n = entry.reduced->boundaryDofCount();
        counts.patterns = entry.reduced->patterns().columns();
        counts.work = entry.patternWork.rows();
    }
    if (entry.boundaryMass.rows() > 0)
        counts.parts |= HAS_BOUNDARY_MASS;
    if (entry.modal)
    {
        counts.parts |= HAS_MODAL_PART;
        counts.modalBoundary = entry.modal->boundaryDofCount();
        counts.modes = entry.modal->modeCount();
    }
    return counts;
}

// The payload words after the counts, false when they do not fit 64 bits.
bool payloadWordsOf(const ReductionCounts& counts, std::uint64_t& words)
{
    std::uint64_t stiffness = 0;
    std::uint64_t work = 0;
    std::uint64_t modal = 0;
    if (!triangleWords(counts.n, stiffness) || !triangleWords(counts.work, work) || !triangleWords(counts.modes, modal))
        return false;
    words = 0;
    if ((counts.parts & HAS_UNIT_PART) != 0)
        words += counts.n + stiffness + counts.n * (counts.patterns + INERTIAL_PATTERN_COUNT) +
                 RIGID_BODY_DOFS * RIGID_BODY_DOFS + work;
    if ((counts.parts & HAS_BOUNDARY_MASS) != 0)
        words += stiffness;
    if ((counts.parts & HAS_MODAL_PART) != 0)
        words += counts.modes + 2 * modal + counts.modalBoundary * counts.modes;
    return true;
}

void writeLowerTriangle(EntryWriter& writer, const linalg::DenseMatrix& matrix)
{
    const std::size_t n = matrix.rows();
    for (std::size_t j = 0; j < n; ++j)
        writer.reals(matrix.data() + j * matrix.leadingDimension() + j, n - j);
}

bool readLowerTriangle(EntryReader& reader, linalg::DenseMatrix& matrix)
{
    const std::size_t n = matrix.rows();
    for (std::size_t j = 0; j < n; ++j)
        if (!reader.reals(matrix.data() + j * matrix.leadingDimension() + j, n - j))
            return false;
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = j + 1; i < n; ++i)
            matrix(j, i) = matrix(i, j);
    return true;
}

} // namespace

std::uint64_t CacheFile::write(const std::filesystem::path& path, const CacheFileHeader& header,
                               const CachedReduction& entry, const std::function<void()>& beforeSection)
{
    const ReductionCounts counts = countsOf(entry);
    const bool hasMass = (counts.parts & HAS_BOUNDARY_MASS) != 0;
    if (counts.parts == 0 || header.kind != CacheEntryKind::Reduction ||
        (hasMass && (entry.reduced == nullptr || entry.boundaryMass.rows() != counts.n ||
                     entry.boundaryMass.columns() != counts.n)))
        throw std::invalid_argument(
            "CacheFile::write: a reduction entry needs its reduced unit (a boundary mass of its boundary) or a modal "
            "part, and a reduction header");
    const auto section = [&]
    {
        if (beforeSection)
            beforeSection();
    };
    return writeEntry(path, header,
                      [&](EntryWriter& writer)
                      {
                          writer.word(counts.parts);
                          writer.word(counts.n);
                          writer.word(counts.patterns);
                          writer.word(counts.work);
                          writer.signedWord(entry.interiorDofCount);
                          writer.real(entry.largestPivotRatio);
                          writer.signedWord(entry.reduced ? entry.reduced->patternStamp() : 0);
                          writer.signedWord(entry.reduced ? entry.reduced->unitId() : 0);
                          writer.word(counts.modalBoundary);
                          writer.word(counts.modes);
                          if (entry.reduced)
                          {
                              const ReducedUnit& reduced = *entry.reduced;
                              const auto n = static_cast<std::size_t>(counts.n);
                              for (const UnitDofLabel& label : reduced.labels())
                                  writer.word(labelWord(label));
                              // K*: the lower triangle by columns (rows j to n − 1 of column j are contiguous).
                              section();
                              writeLowerTriangle(writer, reduced.stiffness());
                              section();
                              const linalg::DenseMatrix& condensed = reduced.patterns();
                              for (std::size_t j = 0; j < condensed.columns(); ++j)
                                  writer.reals(condensed.data() + j * condensed.leadingDimension(), n);
                              const linalg::DenseMatrix& inertial = reduced.inertialPatterns();
                              for (std::size_t j = 0; j < INERTIAL_PATTERN_COUNT; ++j)
                                  writer.reals(inertial.data() + j * inertial.leadingDimension(), n);
                              for (const double value : reduced.rigidBodyMass().values)
                                  writer.real(value);
                              writeLowerTriangle(writer, entry.patternWork);
                          }
                          if (hasMass)
                          {
                              section();
                              writeLowerTriangle(writer, entry.boundaryMass);
                          }
                          if (entry.modal)
                          {
                              section();
                              const ModalReduction& modal = *entry.modal;
                              for (std::size_t k = 0; k < modal.modeCount(); ++k)
                                  writer.real(modal.eigenvalues()(k, 0));
                              writeLowerTriangle(writer, modal.modalStiffness());
                              writeLowerTriangle(writer, modal.modalMass());
                              const linalg::DenseMatrix& coupling = modal.couplingMass();
                              for (std::size_t j = 0; j < coupling.columns(); ++j)
                                  writer.reals(coupling.data() + j * coupling.leadingDimension(), coupling.rows());
                          }
                      });
}

std::uint64_t CacheFile::write(const std::filesystem::path& path, const CacheFileHeader& header,
                               const CachedRecoveryData& entry, const std::function<void()>& beforeSection)
{
    if (header.kind != CacheEntryKind::RecoveryData || entry.eigenvalues.rows() != entry.modes.columns())
        throw std::invalid_argument(
            "CacheFile::write: recovery data needs a recovery-data header and one eigenvalue per mode");
    return writeEntry(path, header,
                      [&](EntryWriter& writer)
                      {
                          writer.word(entry.modes.rows());
                          writer.word(entry.modes.columns());
                          for (std::size_t j = 0; j < entry.modes.columns(); ++j)
                          {
                              if (beforeSection)
                                  beforeSection();
                              writer.reals(entry.modes.data() + j * entry.modes.leadingDimension(), entry.modes.rows());
                          }
                          for (std::size_t k = 0; k < entry.eigenvalues.rows(); ++k)
                              writer.real(entry.eigenvalues(k, 0));
                      });
}

std::uint64_t CacheFile::fileBytes(const CacheFileHeader& header, const CachedReduction& entry)
{
    const ReductionCounts counts = countsOf(entry);
    std::uint64_t payload = 0;
    if (!payloadWordsOf(counts, payload))
        return std::numeric_limits<std::uint64_t>::max();
    // The parts word, the counts, the payload and the checksum.
    const std::uint64_t words = headerWords(header) + 1 + REDUCTION_COUNT_WORDS + payload + 1;
    return words * WORD_BYTES;
}

std::uint64_t CacheFile::fileBytes(const CacheFileHeader& header, const CachedRecoveryData& entry)
{
    constexpr std::uint64_t COUNT_WORDS = 2; // rows, columns
    const std::uint64_t words =
        headerWords(header) + COUNT_WORDS + entry.modes.rows() * entry.modes.columns() + entry.eigenvalues.rows() + 1;
    return words * WORD_BYTES;
}

CacheFileRead<CacheFileHeader> CacheFile::readHeader(const std::filesystem::path& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
        return refused<CacheFileHeader>("cannot be opened (removed by another process, or not readable)");
    EntryReader reader(in);
    CacheFileHeader header;
    if (std::string refusal = readHeaderWords(reader, header); !refusal.empty())
        return refused<CacheFileHeader>(std::move(refusal), reader.bytes());
    CacheFileRead<CacheFileHeader> read;
    read.value = std::move(header);
    read.bytesRead = reader.bytes();
    return read;
}

CacheFileRead<std::shared_ptr<const CachedReduction>> CacheFile::readReduction(const std::filesystem::path& path,
                                                                               const CacheKey& expected,
                                                                               base::MemoryBudget& budget)
{
    using Result = std::shared_ptr<const CachedReduction>;
    const std::uintmax_t fileBytes = fileSizeOf(path);
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
        return refused<Result>("cannot be opened (removed by another process, or not readable)");
    EntryReader reader(in);
    CacheFileHeader header;
    if (std::string refusal = readHeaderWords(reader, header); !refusal.empty())
        return refused<Result>(std::move(refusal), reader.bytes());
    if (header.kind != CacheEntryKind::Reduction)
        return refused<Result>("not a reduction", reader.bytes());
    if (!expected.matches(header.key))
    {
        std::string why = expected.differenceFrom(header.key);
        return refused<Result>(base::concatenate({"another entry: ", why.empty() ? "its unit ID differs" : why}),
                               reader.bytes());
    }
    ReductionCounts counts;
    std::int64_t interiorDofCount = 0;
    double pivotRatio = 0.0;
    std::int64_t patternStamp = 0;
    std::int64_t unitId = 0;
    if (!(reader.word(counts.parts) && reader.word(counts.n) && reader.word(counts.patterns) &&
          reader.word(counts.work) && reader.signedWord(interiorDofCount) && reader.real(pivotRatio) &&
          reader.signedWord(patternStamp) && reader.signedWord(unitId) && reader.word(counts.modalBoundary) &&
          reader.word(counts.modes)))
        return refused<Result>("truncated", reader.bytes());
    // The size the counts give, checked before anything is allocated (a truncated file, or counts that are not
    // O_o_Solver's).
    const bool hasUnit = (counts.parts & HAS_UNIT_PART) != 0;
    const bool hasMass = (counts.parts & HAS_BOUNDARY_MASS) != 0;
    const bool hasModal = (counts.parts & HAS_MODAL_PART) != 0;
    std::uint64_t payload = 0;
    std::uint64_t total = 0;
    const bool isSized = counts.parts != 0 && (counts.parts & ~KNOWN_PARTS) == 0 && (!hasMass || hasUnit) &&
                         counts.n <= MOST_SECTION_ROWS && counts.patterns <= MOST_SECTION_ROWS &&
                         counts.work <= MOST_SECTION_ROWS && counts.modalBoundary <= MOST_SECTION_ROWS &&
                         counts.modes <= MOST_SECTION_ROWS && payloadWordsOf(counts, payload) &&
                         expectedSize(reader.bytes(), payload, total);
    if (!isSized)
        return refused<Result>("corrupted (its sizes are not one O_o_Solver writes)", reader.bytes());
    if (fileBytes != total)
        return refused<Result>(base::concatenate({"truncated (", std::to_string(fileBytes), " bytes, ",
                                                  std::to_string(total), " expected)"}),
                               reader.bytes());

    auto entry = std::make_shared<CachedReduction>();
    entry->equationStamp = header.key.equationStamp;
    const auto size = static_cast<std::size_t>(counts.n);
    if (hasUnit)
    {
        base::BudgetedBuffer<UnitDofLabel> labels(budget, size, "cached unit labels");
        for (std::size_t k = 0; k < size; ++k)
        {
            std::uint64_t word = 0;
            if (!reader.word(word))
                return refused<Result>("truncated", reader.bytes());
            labels[k] = labelOfWord(word);
        }
        linalg::DenseMatrix stiffness(budget, size, size, "cached unit K*");
        if (!readLowerTriangle(reader, stiffness))
            return refused<Result>("truncated", reader.bytes());
        const auto patternCount = static_cast<std::size_t>(counts.patterns);
        linalg::DenseMatrix condensed(budget, size, patternCount, "cached unit patterns");
        for (std::size_t j = 0; j < patternCount; ++j)
            if (!reader.reals(condensed.data() + j * condensed.leadingDimension(), size))
                return refused<Result>("truncated", reader.bytes());
        linalg::DenseMatrix inertial(budget, size, INERTIAL_PATTERN_COUNT, "cached unit inertial patterns");
        for (std::size_t j = 0; j < INERTIAL_PATTERN_COUNT; ++j)
            if (!reader.reals(inertial.data() + j * inertial.leadingDimension(), size))
                return refused<Result>("truncated", reader.bytes());
        base::MatN<6> rigidBodyMass{};
        for (double& value : rigidBodyMass.values)
            if (!reader.real(value))
                return refused<Result>("truncated", reader.bytes());
        const auto workCount = static_cast<std::size_t>(counts.work);
        linalg::DenseMatrix patternWork(budget, workCount, workCount, "cached unit pattern work");
        if (!readLowerTriangle(reader, patternWork))
            return refused<Result>("truncated", reader.bytes());
        // The unit's ID in the plan of this run (the same unit may have had another ID when it was stored).
        static_cast<void>(unitId);
        entry->reduced =
            std::make_shared<const ReducedUnit>(expected.unitId, std::move(labels), std::move(stiffness),
                                                std::move(condensed), patternStamp, std::move(inertial), rigidBodyMass);
        entry->patternWork = std::move(patternWork);
    }
    if (hasMass)
    {
        entry->boundaryMass = linalg::DenseMatrix(budget, size, size, "cached unit boundary mass");
        if (!readLowerTriangle(reader, entry->boundaryMass))
            return refused<Result>("truncated", reader.bytes());
    }
    if (hasModal)
    {
        const auto modes = static_cast<std::size_t>(counts.modes);
        const auto boundary = static_cast<std::size_t>(counts.modalBoundary);
        linalg::DenseMatrix eigenvalues(budget, modes, 1, "cached unit fixed-interface eigenvalues");
        for (std::size_t k = 0; k < modes; ++k)
            if (!reader.real(eigenvalues(k, 0)))
                return refused<Result>("truncated", reader.bytes());
        linalg::DenseMatrix modalStiffness(budget, modes, modes, "cached unit K_qq");
        linalg::DenseMatrix modalMass(budget, modes, modes, "cached unit M_qq");
        if (!readLowerTriangle(reader, modalStiffness) || !readLowerTriangle(reader, modalMass))
            return refused<Result>("truncated", reader.bytes());
        linalg::DenseMatrix coupling(budget, boundary, modes, "cached unit M_aq");
        for (std::size_t j = 0; j < modes; ++j)
            if (!reader.reals(coupling.data() + j * coupling.leadingDimension(), boundary))
                return refused<Result>("truncated", reader.bytes());
        entry->modal = std::make_shared<const ModalReduction>(std::move(eigenvalues), std::move(modalStiffness),
                                                              std::move(coupling), std::move(modalMass));
    }
    if (!reader.isChecksumRight())
        return refused<Result>(reader.isTruncated() ? "truncated" : "corrupted (its checksum differs)", reader.bytes());
    entry->largestPivotRatio = pivotRatio;
    entry->interiorDofCount = interiorDofCount;
    CacheFileRead<Result> read;
    read.value = std::move(entry);
    read.bytesRead = reader.bytes();
    return read;
}

CacheFileRead<std::shared_ptr<const CachedRecoveryData>> CacheFile::readRecoveryData(const std::filesystem::path& path,
                                                                                     const CacheKey& expected,
                                                                                     base::MemoryBudget& budget)
{
    using Result = std::shared_ptr<const CachedRecoveryData>;
    const std::uintmax_t fileBytes = fileSizeOf(path);
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
        return refused<Result>("cannot be opened (removed by another process, or not readable)");
    EntryReader reader(in);
    CacheFileHeader header;
    if (std::string refusal = readHeaderWords(reader, header); !refusal.empty())
        return refused<Result>(std::move(refusal), reader.bytes());
    if (header.kind != CacheEntryKind::RecoveryData)
        return refused<Result>("not recovery data", reader.bytes());
    if (!expected.matches(header.key))
    {
        std::string why = expected.differenceFrom(header.key);
        return refused<Result>(base::concatenate({"another entry: ", why.empty() ? "its unit ID differs" : why}),
                               reader.bytes());
    }
    std::uint64_t rows = 0;
    std::uint64_t columns = 0;
    if (!(reader.word(rows) && reader.word(columns)))
        return refused<Result>("truncated", reader.bytes());
    std::uint64_t total = 0;
    const bool isSized = rows <= MOST_SECTION_ROWS && columns <= MOST_SECTION_ROWS &&
                         expectedSize(reader.bytes(), rows * columns + columns, total);
    if (!isSized)
        return refused<Result>("corrupted (its sizes are not one O_o_Solver writes)", reader.bytes());
    if (fileBytes != total)
        return refused<Result>(base::concatenate({"truncated (", std::to_string(fileBytes), " bytes, ",
                                                  std::to_string(total), " expected)"}),
                               reader.bytes());
    auto entry = std::make_shared<CachedRecoveryData>();
    entry->modes = linalg::DenseMatrix(budget, static_cast<std::size_t>(rows), static_cast<std::size_t>(columns),
                                       "cached fixed-interface modes");
    for (std::size_t j = 0; j < entry->modes.columns(); ++j)
        if (!reader.reals(entry->modes.data() + j * entry->modes.leadingDimension(), entry->modes.rows()))
            return refused<Result>("truncated", reader.bytes());
    entry->eigenvalues =
        linalg::DenseMatrix(budget, static_cast<std::size_t>(columns), 1, "cached fixed-interface eigenvalues");
    for (std::size_t k = 0; k < entry->eigenvalues.rows(); ++k)
        if (!reader.real(entry->eigenvalues(k, 0)))
            return refused<Result>("truncated", reader.bytes());
    if (!reader.isChecksumRight())
        return refused<Result>(reader.isTruncated() ? "truncated" : "corrupted (its checksum differs)", reader.bytes());
    CacheFileRead<Result> read;
    read.value = std::move(entry);
    read.bytesRead = reader.bytes();
    return read;
}

} // namespace oos::reduction
