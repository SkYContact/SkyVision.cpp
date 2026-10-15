// moondream_main.hpp - entry point of the moondream3.1 engine.
//
// moondream3.1 keeps its own argument surface (-m / -t / -v / --region / --serve
// / --batch ...) and its own binary (moondream-cli). The unified front end
// forwards to this entry point so a single command line can drive either model.
#pragma once

// argv_is_utf8: set when the caller already rebuilt argv as UTF-8 (the engine
// otherwise re-reads the process command line on Windows to avoid ANSI mangling).
int moondream_main(int argc, char ** argv, bool argv_is_utf8 = false);
