/**************************************************************************/
/*  streamline.h                                                          */
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

#pragma once

#include "core/object/class_db.h" // IWYU pragma: keep
#include "core/object/object.h"
#include "core/os/mutex.h"
#include "core/os/thread_safe.h"
#include "core/templates/safe_refcount.h"
#include "core/variant/binder_common.h" // IWYU pragma: keep
#include "core/variant/variant.h"
#include "drivers/streamline/streamline_data.h"

class Streamline : public Object {
	GDCLASS(Streamline, Object);
	_THREAD_SAFE_CLASS_
protected:
	static void _bind_methods();
	static Streamline *singleton;
	SafeNumeric<uint32_t> swap_chain_serial;

public:
	static Streamline *get_singleton();
	static void register_singleton();

	void emit_marker(StreamlineMarkerType p_marker);
	void set_parameter(StreamlineParameterType p_parameter_type, const Variant &p_value);
	bool get_capability(StreamlineCapabilityType p_capability_type);

	// Whether Streamline is initialized for Vulkan or D3D12 and not shut down. Before the main device is created, it
	// means Streamline's proxies (D3D12 factory and device, Vulkan interposer) are to be used.
	bool is_active(bool p_d3d12) const;
	// Main device only, right before it is created: whether Streamline supports its GPU, which also enumerates the
	// features available there. When it returns false the driver creates its device without Streamline's proxies,
	// dropping those already in use, then calls shutdown(). Streamline works with one device (ProgrammingGuide.md, 2.5).
	bool select_adapter_vulkan(void *p_physical_device, uint32_t p_vendor_id, const String &p_adapter_name);
	bool select_adapter_d3d12(const void *p_adapter_luid, uint32_t p_luid_size, uint32_t p_vendor_id, const String &p_adapter_name);
	// slShutdown(); afterwards nothing reaches Streamline and every capability is false.
	void shutdown();

	void set_internal_parameter(const char *p_key, void *p_value);
	void *get_internal_parameter(StreamlineInternalParameterType p_internal_parameter_type);

	void update_project_settings();

	// D3D12: bumped when DLSS-G must be loaded or unloaded, which requires a new swap chain.
	// The D3D12 driver recreates a swap chain whose creation serial differs. Any thread.
	uint32_t get_swap_chain_serial() const { return swap_chain_serial.get(); }
	void bump_swap_chain_serial() { swap_chain_serial.increment(); }

	// D3D12, switching the frame generation provider to AMD FSR: returns true when sl.dlss_g is wanted,
	// so the switch must wait for its deferred unload (DLSS-G off meanwhile; the unload then completes
	// the switch). cancel_switch_to_fsr() drops a waiting switch and returns whether there was one.
	bool defer_switch_to_fsr();
	bool cancel_switch_to_fsr();

	// Present markers must carry the token of the frame being presented, which the render thread has already left.
	// 0 (the default on every thread) uses the draw's own token.
	uint64_t get_render_frame_token() const;
	static void set_present_frame_token(uint64_t p_token);

	// Vulkan: latency markers and present use the same VkSwapchainKHR from different threads,
	// which requires external synchronization.
	static Mutex &get_swap_chain_mutex();

	// Vulkan with VK_NV_low_latency2: the driver issues the Reflex sleep instead of slReflexSleep(), so that only
	// vkLatencySleepNV holds the swap chain mutex, not the wait. Returns false if it could not sleep.
	typedef bool (*LatencySleepCallback)(void *p_userdata);
	void set_latency_sleep_callback(LatencySleepCallback p_callback, void *p_userdata);

	Streamline();
	virtual ~Streamline();
};

VARIANT_ENUM_CAST(StreamlineParameterType);
VARIANT_ENUM_CAST(StreamlineCapabilityType);
