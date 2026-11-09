#pragma once

#include <filesystem>
#include <fstream>
#include <string>

namespace oos::base
{

// A file written through a temporary file and renamed onto its target on success (an interruption
// leaves nothing half done). The temporary file sits next to the target, in the same folder, so that the rename stays
// on one file system and replaces the target in one step: a reader sees the old file or the new one, never a part.
// When the object dies without commit() (an error, a cancellation), the temporary file is removed and the target is
// untouched.
//
// The rename is atomic, not durable: a power failure right after it may lose the new content on some file systems;
// the cache, which writes through this class, checks a checksum on reading.
//
// Thread-safety: one object per thread. Ownership: owns the temporary file until commit(); movable, not copyable.
class TemporaryFile
{
public:
    // Creates the temporary file "<target>.tmp-<unique>" and opens it for binary writing. Throws FileError when it
    // cannot be created (missing folder, no permission).
    explicit TemporaryFile(std::filesystem::path target);
    TemporaryFile(const TemporaryFile&) = delete;
    TemporaryFile& operator=(const TemporaryFile&) = delete;
    TemporaryFile(TemporaryFile&& other) noexcept;
    TemporaryFile& operator=(TemporaryFile&& other) noexcept;
    ~TemporaryFile();

    // The stream to write the content to. Precondition: commit() has not been called.
    [[nodiscard]] std::ofstream& stream() noexcept { return m_stream; }

    [[nodiscard]] const std::filesystem::path& targetPath() const noexcept { return m_target; }
    [[nodiscard]] const std::filesystem::path& temporaryPath() const noexcept { return m_temporary; }
    [[nodiscard]] bool isCommitted() const noexcept { return m_isCommitted; }

    // Flushes and closes the stream, then renames the temporary file onto the target, replacing a file already there.
    // Throws FileError when a write failed or the rename fails; the temporary file is then removed. A second call after
    // a successful one does nothing.
    void commit();

private:
    void discard() noexcept;

    std::filesystem::path m_target;
    std::filesystem::path m_temporary;
    std::ofstream m_stream;
    bool m_isCommitted = false;
};

} // namespace oos::base
