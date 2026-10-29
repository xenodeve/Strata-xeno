"""Shared offline Hub diagnostic contracts; no tensor runtime import."""
import ctypes
from tools.qwenputer.memory import ExtendedLimit,job_fields

def validate_launcher_env(env):
    # Actual selected launcher, strata-hub.bat:131-132. Profile JSON alone does
    # not carry these settings; refusing here avoids allocating on the wrong GPU.
    if env.get('CUDA_VISIBLE_DEVICES')!='1,0' or env.get('CUDA_DEVICE_ORDER','')!='':
        raise ValueError('output collector requires selected Hub launcher CUDA_VISIBLE_DEVICES=1,0 and default CUDA_DEVICE_ORDER')


def current_job():
    info=ExtendedLimit();query=ctypes.WinDLL('kernel32',use_last_error=True).QueryInformationJobObject
    query.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_uint32,ctypes.c_void_p]
    if not query(None,9,ctypes.byref(info),ctypes.sizeof(info),None):raise ctypes.WinError(ctypes.get_last_error())
    value=job_fields(info)
    if value['limit_flags']&0x300:raise RuntimeError('output collection must not have a private/job commit cap')
    return value
