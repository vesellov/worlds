/*
 * eitrace: which functions of game.exe run, phase by phase.
 *
 * A small Win32 debugger. It starts game.exe of the folder given as the first argument (default the current
 * folder; the other arguments go to the game) as its debuggee, puts int3 on the first
 * byte of every function listed in eitrace_addrs.txt and, when one is hit, notes it, puts the byte back and
 * lets the thread go on. So every function costs one debug event per window, not per call.
 *
 * The files sit next to eitrace.exe:
 *   eitrace_addrs.txt   the addresses, one hex number per line (written by tools/func_coverage.py)
 *   eitrace_phase.txt   the name of the current phase, written by the scenario driver; on a change of
 *                       the name every function is armed again
 *   eitrace_out.txt     the result, rewritten every few seconds and at the exit of the game:
 *                         P <phase index> <start ms> <name>
 *                         H <address> <phase index> <first ms> <windows>
 *                       windows is how many windows of WINDOW_MS (env EITRACE_WINDOW_MS) of the phase
 *                       saw the function: the functions hit in a window are armed again at its end
 *   eitrace.log         what the tracer did
 *
 * Under the Wine of the game (no Z: there): C:\eitrace\eitrace.exe K:\game_info\data
 *
 * Build: i686-w64-mingw32-gcc -O2 -o eitrace.exe eitrace.c (make eitrace)
 */

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_FUNCS 20000
#define MAX_PHASES 64
#define PHASE_NAME 64
#define TEXT_START 0x401000
#define WRITE_MS 5000
#define PHASE_POLL_MS 100

static DWORD addrs[MAX_FUNCS];
static int n_funcs;
static unsigned char armed[MAX_FUNCS];
static unsigned char hit_now[MAX_FUNCS];
static int hit_list[MAX_FUNCS];
static int n_hit_list;

static char phase_names[MAX_PHASES][PHASE_NAME];
static DWORD phase_start[MAX_PHASES];
static int n_phases;
static int cur_phase;
static DWORD first_ms[MAX_PHASES][MAX_FUNCS];   /* ms + 1, 0 is never */
static unsigned short windows_hit[MAX_PHASES][MAX_FUNCS];

static unsigned char *text_orig;
static unsigned char *text_shadow;
static DWORD text_size;

static HANDLE process;
static DWORD t0;
static DWORD window_ms = 2000;
static char dir_path[MAX_PATH];
static char addrs_file_path[MAX_PATH + 32];
static char phase_file_path[MAX_PATH + 32];
static char out_file_path[MAX_PATH + 32];
static char log_file_path[MAX_PATH + 32];
static FILE *log_file;

static void logf_(const char *fmt, ...)
{
    va_list ap;
    if (!log_file)
        return;
    fprintf(log_file, "%8lu ", (unsigned long)(GetTickCount() - t0));
    va_start(ap, fmt);
    vfprintf(log_file, fmt, ap);
    va_end(ap);
    fputc('\n', log_file);
    fflush(log_file);
}

static int cmp_dword(const void *a, const void *b)
{
    DWORD x = *(const DWORD *)a, y = *(const DWORD *)b;
    return x < y ? -1 : x > y;
}

static int find_func(DWORD address)
{
    int lo = 0, hi = n_funcs - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (addrs[mid] == address)
            return mid;
        if (addrs[mid] < address)
            lo = mid + 1;
        else
            hi = mid - 1;
    }
    return -1;
}

static int load_addrs(void)
{
    char line[64];
    FILE *f = fopen(addrs_file_path, "r");
    if (!f)
        return 0;
    while (fgets(line, sizeof(line), f) && n_funcs < MAX_FUNCS) {
        DWORD a = strtoul(line, NULL, 16);
        if (a >= TEXT_START)
            addrs[n_funcs++] = a;
    }
    fclose(f);
    qsort(addrs, n_funcs, sizeof(DWORD), cmp_dword);
    return n_funcs;
}

static void write_byte(DWORD address, unsigned char value)
{
    SIZE_T done = 0;
    WriteProcessMemory(process, (LPVOID)(ULONG_PTR)address, &value, 1, &done);
    text_shadow[address - TEXT_START] = value;
}

/* every function gets its int3 again: one write of the whole .text from the shadow copy */
static void arm_all(void)
{
    SIZE_T done = 0;
    int i;
    for (i = 0; i < n_funcs; i++) {
        DWORD off = addrs[i] - TEXT_START;
        if (off < text_size) {
            text_shadow[off] = 0xCC;
            armed[i] = 1;
        }
        hit_now[i] = 0;
    }
    n_hit_list = 0;
    if (!WriteProcessMemory(process, (LPVOID)TEXT_START, text_shadow, text_size, &done))
        logf_("arm_all: WriteProcessMemory failed %lu", GetLastError());
    FlushInstructionCache(process, (LPVOID)TEXT_START, text_size);
}

/* the end of a window: the functions hit in it get their int3 back */
static void arm_hit(void)
{
    int k;
    for (k = 0; k < n_hit_list; k++) {
        int i = hit_list[k];
        hit_now[i] = 0;
        if (!armed[i]) {
            write_byte(addrs[i], 0xCC);
            armed[i] = 1;
        }
    }
    n_hit_list = 0;
    FlushInstructionCache(process, (LPVOID)TEXT_START, text_size);
}

static void set_phase(const char *name)
{
    int i;
    for (i = 0; i < n_phases; i++)
        if (strcmp(phase_names[i], name) == 0)
            break;
    if (i == n_phases) {
        if (n_phases >= MAX_PHASES) {
            logf_("too many phases, %s is not added", name);
            return;
        }
        strncpy(phase_names[i], name, PHASE_NAME - 1);
        phase_start[i] = GetTickCount() - t0;
        n_phases++;
    }
    cur_phase = i;
    logf_("phase %d %s", i, name);
}

static void poll_phase(int arm)
{
    char line[PHASE_NAME];
    char *p;
    FILE *f = fopen(phase_file_path, "r");
    if (!f)
        return;
    line[0] = 0;
    if (!fgets(line, sizeof(line), f))
        line[0] = 0;
    fclose(f);
    for (p = line; *p; p++)
        if (*p == '\r' || *p == '\n' || *p == ' ' || *p == '\t')
            *p = 0;
    if (!line[0] || (n_phases && strcmp(line, phase_names[cur_phase]) == 0))
        return;
    set_phase(line);
    if (arm)
        arm_all();
}

static void write_out(void)
{
    char tmp_path[MAX_PATH + 40];
    FILE *f;
    int ph, i;
    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", out_file_path);
    f = fopen(tmp_path, "w");
    if (!f)
        return;
    fprintf(f, "# eitrace window_ms %lu now %lu funcs %d\n", (unsigned long)window_ms,
            (unsigned long)(GetTickCount() - t0), n_funcs);
    for (ph = 0; ph < n_phases; ph++)
        fprintf(f, "P %d %lu %s\n", ph, (unsigned long)phase_start[ph], phase_names[ph]);
    for (ph = 0; ph < n_phases; ph++)
        for (i = 0; i < n_funcs; i++)
            if (first_ms[ph][i])
                fprintf(f, "H %06lx %d %lu %u\n", (unsigned long)addrs[i], ph,
                        (unsigned long)(first_ms[ph][i] - 1), windows_hit[ph][i]);
    fclose(f);
    MoveFileExA(tmp_path, out_file_path, MOVEFILE_REPLACE_EXISTING);
}

static int start_tracing(void)
{
    SIZE_T done = 0;
    DWORD old;
    text_size = 0x33a000;
    text_orig = malloc(text_size);
    text_shadow = malloc(text_size);
    if (!ReadProcessMemory(process, (LPVOID)TEXT_START, text_orig, text_size, &done)) {
        logf_("ReadProcessMemory of .text failed %lu (read %lu)", GetLastError(), (unsigned long)done);
        return 0;
    }
    memcpy(text_shadow, text_orig, text_size);
    if (!VirtualProtectEx(process, (LPVOID)TEXT_START, text_size, PAGE_EXECUTE_READWRITE, &old))
        logf_("VirtualProtectEx failed %lu", GetLastError());
    arm_all();
    logf_("armed %d functions", n_funcs);
    return 1;
}

/* returns the continue status */
static DWORD on_breakpoint(DEBUG_EVENT *ev, int *seen_loader)
{
    DWORD address = (DWORD)(ULONG_PTR)ev->u.Exception.ExceptionRecord.ExceptionAddress;
    int i = find_func(address);
    HANDLE thread;
    CONTEXT ctx;
    if (i < 0) {
        if (!*seen_loader) {
            *seen_loader = 1;
            return DBG_CONTINUE;
        }
        return DBG_EXCEPTION_NOT_HANDLED;
    }
    if (armed[i]) {
        write_byte(address, text_orig[address - TEXT_START]);
        FlushInstructionCache(process, (LPVOID)(ULONG_PTR)address, 1);
        armed[i] = 0;
    }
    if (!hit_now[i]) {
        hit_now[i] = 1;
        hit_list[n_hit_list++] = i;
        if (!first_ms[cur_phase][i])
            first_ms[cur_phase][i] = GetTickCount() - t0 + 1;
        if (windows_hit[cur_phase][i] < 65535)
            windows_hit[cur_phase][i]++;
    }
    /* the thread goes back to the first byte of the function, now the original one */
    thread = OpenThread(THREAD_GET_CONTEXT | THREAD_SET_CONTEXT, FALSE, ev->dwThreadId);
    if (thread) {
        ctx.ContextFlags = CONTEXT_CONTROL;
        if (GetThreadContext(thread, &ctx)) {
            ctx.Eip = address;
            SetThreadContext(thread, &ctx);
        } else {
            logf_("GetThreadContext failed %lu", GetLastError());
        }
        CloseHandle(thread);
    } else {
        logf_("OpenThread %lu failed %lu", ev->dwThreadId, GetLastError());
    }
    return DBG_CONTINUE;
}

int main(int argc, char **argv)
{
    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    DEBUG_EVENT ev;
    char cmd[1024];
    char *slash;
    const char *env;
    const char *game_dir;
    int seen_loader = 0;
    int running = 1;
    DWORD last_poll = 0, last_window = 0, last_write = 0;
    int i;

    t0 = GetTickCount();
    GetModuleFileNameA(NULL, dir_path, sizeof(dir_path));
    slash = strrchr(dir_path, '\\');
    if (slash)
        *slash = 0;
    snprintf(addrs_file_path, sizeof(addrs_file_path), "%s\\eitrace_addrs.txt", dir_path);
    snprintf(phase_file_path, sizeof(phase_file_path), "%s\\eitrace_phase.txt", dir_path);
    snprintf(out_file_path, sizeof(out_file_path), "%s\\eitrace_out.txt", dir_path);
    snprintf(log_file_path, sizeof(log_file_path), "%s\\eitrace.log", dir_path);
    log_file = fopen(log_file_path, "w");
    env = getenv("EITRACE_WINDOW_MS");
    if (env && atoi(env) > 0)
        window_ms = atoi(env);

    if (!load_addrs()) {
        logf_("no addresses in %s", addrs_file_path);
        return 1;
    }
    logf_("%d addresses, window %lu ms", n_funcs, (unsigned long)window_ms);
    poll_phase(0);
    if (!n_phases)
        set_phase("boot");

    game_dir = argc > 1 ? argv[1] : ".";
    snprintf(cmd, sizeof(cmd), "\"%s\\game.exe\"", game_dir);
    for (i = 2; i < argc; i++) {
        strncat(cmd, " ", sizeof(cmd) - strlen(cmd) - 1);
        strncat(cmd, argv[i], sizeof(cmd) - strlen(cmd) - 1);
    }
    memset(&si, 0, sizeof(si));
    si.cb = sizeof(si);
    if (!CreateProcessA(NULL, cmd, NULL, NULL, FALSE, DEBUG_ONLY_THIS_PROCESS, NULL, game_dir, &si, &pi)) {
        logf_("CreateProcess %s failed %lu", cmd, GetLastError());
        return 1;
    }
    process = pi.hProcess;
    logf_("started %s pid %lu", cmd, pi.dwProcessId);

    while (running) {
        DWORD now;
        if (WaitForDebugEvent(&ev, PHASE_POLL_MS)) {
            DWORD status = DBG_CONTINUE;
            switch (ev.dwDebugEventCode) {
            case CREATE_PROCESS_DEBUG_EVENT:
                if (ev.u.CreateProcessInfo.hFile)
                    CloseHandle(ev.u.CreateProcessInfo.hFile);
                if (!start_tracing())
                    logf_("tracing is off");
                break;
            case LOAD_DLL_DEBUG_EVENT:
                if (ev.u.LoadDll.hFile)
                    CloseHandle(ev.u.LoadDll.hFile);
                break;
            case EXCEPTION_DEBUG_EVENT:
                if (ev.u.Exception.ExceptionRecord.ExceptionCode == EXCEPTION_BREAKPOINT)
                    status = on_breakpoint(&ev, &seen_loader);
                else
                    status = DBG_EXCEPTION_NOT_HANDLED;
                break;
            case EXIT_PROCESS_DEBUG_EVENT:
                logf_("game exited, code %lu", ev.u.ExitProcess.dwExitCode);
                running = 0;
                break;
            default:
                break;
            }
            ContinueDebugEvent(ev.dwProcessId, ev.dwThreadId, status);
        }
        if (!running || !text_orig)
            continue;
        now = GetTickCount();
        if (now - last_poll >= PHASE_POLL_MS) {
            last_poll = now;
            poll_phase(1);
        }
        if (now - last_window >= window_ms) {
            last_window = now;
            arm_hit();
        }
        if (now - last_write >= WRITE_MS) {
            last_write = now;
            write_out();
        }
    }
    write_out();
    logf_("done");
    return 0;
}
