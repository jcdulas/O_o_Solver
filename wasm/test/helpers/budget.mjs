// The time limits of the fast suite. A developer's machine keeps the limits below. A machine that is slower or shared
// (a runner of the continuous integration) gives the factor it needs with OOS_TEST_TIME_FACTOR: every case keeps its
// watchdog and its ceiling, multiplied by that factor, so that a case that hangs is still interrupted.

const BASE_CASE_BUDGET_MS = 900;
const BASE_CASE_CEILING_MS = 1000;

function readTimeFactor() {
  const text = process.env.OOS_TEST_TIME_FACTOR;
  if (text === undefined || text === "") return 1;
  const factor = Number(text);
  if (!Number.isFinite(factor) || factor < 1)
    throw new Error(`OOS_TEST_TIME_FACTOR must be a number of at least 1, found '${text}'`);
  return factor;
}

export const TIME_FACTOR = readTimeFactor();
// Keep a tenth of the ceiling for message delivery and test-runner bookkeeping.
export const CASE_BUDGET_MS = BASE_CASE_BUDGET_MS * TIME_FACTOR;
export const CASE_CEILING_MS = BASE_CASE_CEILING_MS * TIME_FACTOR;
