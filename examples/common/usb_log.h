// usb_log.h - non-blocking USB serial log lines for the examples.
//
// libDaisy's Logger (hw.PrintLine) switches to blocking transmits once a
// terminal has connected: if the terminal is then closed without unplugging
// USB, the next PrintLine waits forever. With a main-loop stage that stalls
// DoBackgroundWork and the reverb tail drops out (hundreds of underruns);
// here a line is dropped instead when the host is not reading.
//
// Call hw.StartLog(false) once first (it brings up the USB CDC port).

#pragma once

#include <cstdarg>
#include <cstdio>

#include "daisy_seed.h"

namespace usblog
{

// The USB peripheral reads the buffer after Transmit returns, so a submitted
// buffer must not be touched until its transfer is done: two static
// buffers, always writing the one that was not submitted last.
inline char   buf[2][192];
inline size_t inflight = 1;

inline bool Line(daisy::DaisySeed& hw, const char* fmt, ...)
{
    const size_t w = inflight ^ 1;
    char*        b = buf[w];
    va_list      va;
    va_start(va, fmt);
    int n = vsnprintf(b, sizeof(buf[0]) - 2, fmt, va);
    va_end(va);
    if(n < 0)
        return false;
    if(n > (int)sizeof(buf[0]) - 3)
        n = (int)sizeof(buf[0]) - 3;
    b[n++] = '\r';
    b[n++] = '\n';
    if(hw.usb_handle.TransmitInternal((uint8_t*)b, (size_t)n)
       != daisy::UsbHandle::Result::OK)
        return false; // busy (host not reading): drop the line
    inflight = w;
    return true;
}

} // namespace usblog
