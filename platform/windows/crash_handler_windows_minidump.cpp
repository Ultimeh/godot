/**************************************************************************/
/*  crash_handler_windows_minidump.cpp                                    */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "crash_handler_windows.h"

#include "core/error/error_macros.h"
#include "core/os/os.h"
#include "core/string/print_string.h"

// Some versions of imagehlp.dll lack the proper packing directives themselves
// so we need to do it.
#pragma pack(push, before_dbghelp, 8)
#include <dbghelp.h>
#pragma pack(pop, before_dbghelp)

// Minidumps of crashes, on any thread and in every build type, written with MiniDumpWriteDump (Microsoft's
// documented practice) from two places:
// - an unhandled exception filter, for any exception nothing handles. The debug-only backtrace of the main
//   thread (CrashHandlerException) runs before it and is unchanged;
// - a vectored exception handler, for hardware faults (and CRASH_NOW's trap) in the engine's own code only. The
//   .NET runtime does not let them reach the filter when they cross managed frames (a C# script calling the
//   engine): it turns them into a fatal AccessViolationException and fails fast. The engine never handles a
//   fault in its own code, so such a first-chance exception is always a crash.
//
// Everything they need is prepared when the dumps are enabled, so that a crash (heap corruption included)
// allocates nothing: dbghelp is loaded, the dump directory is converted, and the dump is written by a thread
// created in advance, which also keeps a stack overflow from leaving the writer without stack.

typedef BOOL(WINAPI *MiniDumpWriteDumpFunc)(HANDLE, DWORD, HANDLE, MINIDUMP_TYPE, PMINIDUMP_EXCEPTION_INFORMATION, PMINIDUMP_USER_STREAM_INFORMATION, PMINIDUMP_CALLBACK_INFORMATION);

// Small dumps (a few hundred KB to a few MB): the stacks of every thread, the memory their locals point to,
// the thread times and the unloaded modules. Code is read from the binaries and their PDBs when opened.
static const MINIDUMP_TYPE MINIDUMP_FLAGS = (MINIDUMP_TYPE)(MiniDumpWithIndirectlyReferencedMemory | MiniDumpWithThreadInfo | MiniDumpWithUnloadedModules);

// Longest directory: the file name appended at crash time ("\YYYY-MM-DD_HH-MM-SS_<pid>.dmp") must still fit.
static const int MINIDUMP_DIR_MAX_LENGTH = MAX_PATH - 40;

static MiniDumpWriteDumpFunc minidump_write_dump = nullptr;
static LPTOP_LEVEL_EXCEPTION_FILTER minidump_previous_filter = nullptr;
static WCHAR minidump_path[MAX_PATH] = {};
static int minidump_dir_length = 0;
static uintptr_t minidump_engine_begin = 0;
static uintptr_t minidump_engine_end = 0;

static HANDLE minidump_writer_thread = nullptr;
static HANDLE minidump_request_event = nullptr;
static HANDLE minidump_done_event = nullptr;
static EXCEPTION_POINTERS *minidump_exception = nullptr;
static DWORD minidump_thread_id = 0;
static BOOL minidump_written = FALSE;
static volatile LONG minidump_requested = 0;
static volatile LONG minidump_previous_filter_called = 0;

static DWORD WINAPI _minidump_writer(LPVOID p_userdata) {
	WaitForSingleObject(minidump_request_event, INFINITE);

	HANDLE file = CreateFileW(minidump_path, GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
	if (file != INVALID_HANDLE_VALUE) {
		MINIDUMP_EXCEPTION_INFORMATION exception_info = {};
		exception_info.ThreadId = minidump_thread_id;
		exception_info.ExceptionPointers = minidump_exception;
		exception_info.ClientPointers = FALSE;
		minidump_written = minidump_write_dump(GetCurrentProcess(), GetCurrentProcessId(), file, MINIDUMP_FLAGS, &exception_info, nullptr, nullptr);
		CloseHandle(file);
		if (!minidump_written) {
			DeleteFileW(minidump_path);
		}
	}

	SetEvent(minidump_done_event);
	return 0;
}

// Writes one dump per process: a second exception (another thread crashing meanwhile, or the same crash
// reaching the filter after the vectored handler) writes nothing.
static void _minidump_write(EXCEPTION_POINTERS *p_exception) {
	OS *os = OS::get_singleton();
	if (os != nullptr && (os->is_disable_crash_handler() || os->is_crash_handler_silent())) {
		return;
	}
	if (InterlockedExchange(&minidump_requested, 1) != 0) {
		return;
	}

	// The directory part of minidump_path is filled by enable_minidumps().
	minidump_path[minidump_dir_length] = L'\0';
	CreateDirectoryW(minidump_path, nullptr);

	SYSTEMTIME time;
	GetLocalTime(&time);
	WCHAR *file_name = minidump_path + minidump_dir_length;
	wsprintfW(file_name, L"\\%04u-%02u-%02u_%02u-%02u-%02u.dmp", time.wYear, time.wMonth, time.wDay, time.wHour, time.wMinute, time.wSecond);
	if (GetFileAttributesW(minidump_path) != INVALID_FILE_ATTRIBUTES) {
		wsprintfW(file_name, L"\\%04u-%02u-%02u_%02u-%02u-%02u_%lu.dmp", time.wYear, time.wMonth, time.wDay, time.wHour, time.wMinute, time.wSecond, GetCurrentProcessId());
	}

	minidump_exception = p_exception;
	minidump_thread_id = GetCurrentThreadId();
	SetEvent(minidump_request_event);
	// Bounded, so that a writer that cannot finish never turns the crash into a hang.
	const bool finished = WaitForSingleObject(minidump_done_event, 120000) == WAIT_OBJECT_0;

	// Logging allocates: it comes after the dump, so that a corrupted heap can only cost the message.
	if (os != nullptr) {
		if (finished && minidump_written) {
			print_error(vformat("Crash minidump written to: %s", String::utf16((const char16_t *)minidump_path)));
		} else {
			print_error(vformat("Writing the crash minidump failed: %s", String::utf16((const char16_t *)minidump_path)));
		}
	}
}

static LONG WINAPI _minidump_vectored_handler(EXCEPTION_POINTERS *p_exception) {
	switch (p_exception->ExceptionRecord->ExceptionCode) {
		case EXCEPTION_ACCESS_VIOLATION:
		case EXCEPTION_IN_PAGE_ERROR:
		case EXCEPTION_ILLEGAL_INSTRUCTION:
		case EXCEPTION_PRIV_INSTRUCTION:
		case EXCEPTION_INT_DIVIDE_BY_ZERO:
		case EXCEPTION_ARRAY_BOUNDS_EXCEEDED:
		case EXCEPTION_DATATYPE_MISALIGNMENT:
		case EXCEPTION_STACK_OVERFLOW:
		case EXCEPTION_BREAKPOINT: // GENERATE_TRAP() of the CRASH_* macros.
			break;
		default:
			return EXCEPTION_CONTINUE_SEARCH;
	}
	const uintptr_t address = (uintptr_t)p_exception->ExceptionRecord->ExceptionAddress;
	if (address < minidump_engine_begin || address >= minidump_engine_end || IsDebuggerPresent()) {
		return EXCEPTION_CONTINUE_SEARCH;
	}
	_minidump_write(p_exception);
	return EXCEPTION_CONTINUE_SEARCH;
}

static LONG WINAPI _minidump_exception_filter(EXCEPTION_POINTERS *p_exception) {
	_minidump_write(p_exception);

	// The filter we were installed in front of runs next (the .NET runtime's, for instance), once: it may itself
	// call the filter it replaced, which can be this one.
	if (minidump_previous_filter != nullptr && InterlockedExchange(&minidump_previous_filter_called, 1) == 0) {
		return minidump_previous_filter(p_exception);
	}
	return EXCEPTION_CONTINUE_SEARCH;
}

static void _minidump_install_filter() {
	const LPTOP_LEVEL_EXCEPTION_FILTER previous = SetUnhandledExceptionFilter(_minidump_exception_filter);
	if (previous == _minidump_exception_filter) {
		return;
	}
	if (minidump_previous_filter != nullptr) {
		// Something installed its own filter since the last call, and does not call this one.
		HMODULE module = nullptr;
		WCHAR module_path[MAX_PATH] = {};
		if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCWSTR)(void *)previous, &module)) {
			GetModuleFileNameW(module, module_path, MAX_PATH);
		}
		print_verbose(vformat("Crash minidumps: the unhandled exception filter was replaced by %s; installed again in front of it.", String::utf16((const char16_t *)module_path)));
	}
	minidump_previous_filter = previous;
}

void CrashHandler::enable_minidumps(const String &p_dir) {
	if (minidump_writer_thread != nullptr) {
		// Called again once runtimes that install their own filter are loaded (the .NET runtime does,
		// without calling the previous one for native exceptions): put this one back in front.
		_minidump_install_filter();
		return;
	}

	const Char16String dir = p_dir.replace_char('/', '\\').utf16();
	ERR_FAIL_COND_MSG(dir.length() > MINIDUMP_DIR_MAX_LENGTH, vformat("Crash minidumps are disabled: the path \"%s\" is too long.", p_dir));

	// From System32 only, never from the game's folder.
	HMODULE dbghelp = LoadLibraryExW(L"dbghelp.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
	ERR_FAIL_NULL_MSG(dbghelp, "Crash minidumps are disabled: dbghelp.dll could not be loaded.");
	minidump_write_dump = (MiniDumpWriteDumpFunc)(void *)GetProcAddress(dbghelp, "MiniDumpWriteDump");
	ERR_FAIL_NULL_MSG(minidump_write_dump, "Crash minidumps are disabled: MiniDumpWriteDump was not found in dbghelp.dll.");

	// The engine's image: the executable, or the library when built as libgodot.
	HMODULE engine = nullptr;
	ERR_FAIL_COND_MSG(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCWSTR)(void *)&_minidump_vectored_handler, &engine), "Crash minidumps are disabled: the engine's module was not found.");
	const IMAGE_NT_HEADERS *headers = (const IMAGE_NT_HEADERS *)((const uint8_t *)engine + ((const IMAGE_DOS_HEADER *)engine)->e_lfanew);
	minidump_engine_begin = (uintptr_t)engine;
	minidump_engine_end = minidump_engine_begin + headers->OptionalHeader.SizeOfImage;

	memcpy(minidump_path, dir.get_data(), dir.length() * sizeof(WCHAR));
	minidump_dir_length = dir.length();
	minidump_path[minidump_dir_length] = L'\0';

	minidump_request_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
	minidump_done_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
	ERR_FAIL_COND_MSG(minidump_request_event == nullptr || minidump_done_event == nullptr, "Crash minidumps are disabled: their events could not be created.");
	minidump_writer_thread = CreateThread(nullptr, 0, _minidump_writer, nullptr, 0, nullptr);
	ERR_FAIL_NULL_MSG(minidump_writer_thread, "Crash minidumps are disabled: their writer thread could not be created.");

	// First in the list, so that it runs before the .NET runtime's own vectored handler.
	AddVectoredExceptionHandler(1, _minidump_vectored_handler);
	_minidump_install_filter();
}
