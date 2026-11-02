"""Breach: reach the Task Scheduler, from where a task would outlive the box. This calls ITaskService::Connect, the first
thing that talks to the service, and creates nothing. Inside a box the client library fails first: this call gives
0x80040154 or dies with heap corruption 0xC0000374, as schtasks.exe does. So it shows no connection was made, not that the
service said no; test_escape.py gets that from the service's own pipe."""
import ctypes
import sys
import uuid

CLSCTX_INPROC_SERVER = 1
TASK_SCHEDULER = uuid.UUID("0F87369F-A4E5-4CFC-BD3E-73E6154572DD").bytes_le
ITASK_SERVICE = uuid.UUID("2FABA4C7-4DA9-4013-9697-20CC3FD40F85").bytes_le
CONNECT_SLOT = 10    # after IUnknown (3), IDispatch (4), GetFolder, GetRunningTasks, NewTask


class Variant(ctypes.Structure):
    _fields_ = [("data", ctypes.c_ubyte * 24)]


ole = ctypes.windll.ole32
ole.CoInitializeEx(None, 2)
service = ctypes.c_void_p()
status = ole.CoCreateInstance(TASK_SCHEDULER, None, CLSCTX_INPROC_SERVER, ITASK_SERVICE, ctypes.byref(service)) & 0xFFFFFFFF
if status:
    print(f"refused: CoCreateInstance, {status:#010x}")
    sys.exit(10)
table = ctypes.cast(service, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
connect = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *[ctypes.POINTER(Variant)] * 4)(table[CONNECT_SLOT])
empty = [Variant() for _ in range(4)]
status = connect(service, *[ctypes.byref(item) for item in empty]) & 0xFFFFFFFF
if status:
    print(f"refused: ITaskService::Connect, {status:#010x}")
    sys.exit(10)
print("connected to the Task Scheduler")
