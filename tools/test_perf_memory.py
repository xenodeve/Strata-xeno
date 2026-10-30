"""ABBA memory semantics with fake Win32 APIs; never query live admission."""
import ctypes
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from tools.qwenputer import memory as telemetry

path=Path(__file__).resolve().parents[1]/'tests/xeno/perf/ab.py'
spec=importlib.util.spec_from_file_location('strata_perf_ab_memory_test',path)
ab=importlib.util.module_from_spec(spec);spec.loader.exec_module(ab)


class PerfMemory(unittest.TestCase):
    def libraries(self,process_limit_gib=42,process_available_gib=2,api_success=True):
        calls=[]
        def global_status(pointer):
            calls.append('GlobalMemoryStatusEx');m=pointer._obj
            m.ullTotalPhys=48*1024**3;m.ullAvailPhys=20*1024**3
            m.ullTotalPageFile=process_limit_gib*1024**3;m.ullAvailPageFile=process_available_gib*1024**3
            return 1
        def performance(pointer,size):
            calls.append('GetPerformanceInfo');p=pointer._obj
            self.assertEqual(size,ctypes.sizeof(p));self.assertEqual(p.cb,size)
            p.PageSize=4096;p.CommitTotal=50*1024**3//4096;p.CommitLimit=82*1024**3//4096
            p.PhysicalTotal=48*1024**3//4096;p.PhysicalAvailable=20*1024**3//4096
            return int(api_success)
        kernel=SimpleNamespace(GlobalMemoryStatusEx=global_status)
        psapi=SimpleNamespace(GetPerformanceInfo=performance)
        return SimpleNamespace(kernel32=kernel),lambda name,**kwargs:psapi if name=='psapi' else kernel,calls

    def test_process_commit_allowance_is_not_system_commitment(self):
        for limit,available in ((42,2),(82,32)):
            windll,dll,calls=self.libraries(limit,available)
            with self.subTest(limit=limit),patch.object(telemetry,'_libraries',return_value=(windll.kernel32,dll('psapi'))):
                self.assertEqual(ab.system_memory(),(28.,20.,50.))
                self.assertEqual(calls,['GetPerformanceInfo'])

    def test_api_failure_is_not_reported_as_zero_commit(self):
        windll,dll,_=self.libraries(api_success=False)
        with patch.object(telemetry,'_libraries',return_value=(windll.kernel32,dll('psapi'))),\
             patch.object(ctypes,'get_last_error',lambda:5,create=True),\
             patch.object(ctypes,'WinError',lambda code:OSError(code,'fixture Win32 failure'),create=True):
            with self.assertRaises(OSError):ab.system_memory()

    def test_fake_api_semantics_do_not_require_a_windows_host(self):
        windll,dll,calls=self.libraries()
        with patch.object(telemetry.os,'name','posix'),patch.object(telemetry,'_libraries',return_value=(windll.kernel32,dll('psapi'))):
            self.assertEqual(ab.system_memory(),(28.,20.,50.))
        self.assertEqual(calls,['GetPerformanceInfo'])

    def test_sampler_failure_stays_unavailable_instead_of_a_valid_zero_peak(self):
        proc=SimpleNamespace(pid=17,poll=lambda:None);samples=[];errors=[]
        with patch.object(ab,'system_memory',side_effect=OSError('fake telemetry failure')):
            ab.collect_memory(proc,samples.append,errors)
        self.assertEqual(samples,[]);self.assertEqual(len(errors),1)
        line=ab.memory_peak_line([0,0,0,0],0,errors)
        self.assertIn('memory unavailable',line);self.assertNotIn('memory peak',line)

    def test_successful_sampler_keeps_the_existing_peak_parser_contract(self):
        remaining=[None,0];proc=SimpleNamespace(pid=17,poll=lambda:remaining.pop(0))
        samples=[];errors=[]
        with patch.object(ab,'system_memory',return_value=(28.,20.,50.)),patch.object(ab,'process_memory',return_value=(3.,2.)),patch.object(ab.time,'sleep'):
            ab.collect_memory(proc,samples.append,errors)
        self.assertEqual(samples,[(28.,20.,50.,3.,2.)]);self.assertEqual(errors,[])
        line=ab.memory_peak_line([28.,50.,3.,2.],1,errors)
        _,values=ab.parse_run('output : 1 2 3\n'+line)
        self.assertEqual(values['sys_commit_gib'],50.)
        self.assertEqual(values['proc_private_gib'],3.)

    def test_mem_trace_preserves_logs_and_fails_when_memory_sampling_fails(self):
        module_spec=importlib.util.spec_from_file_location('strata_mem_trace_test',path.with_name('mem_trace.py'))
        module=importlib.util.module_from_spec(module_spec)
        with patch.object(sys,'argv',['mem_trace.py']),patch('subprocess.Popen',side_effect=AssertionError('no launch during import')):
            module_spec.loader.exec_module(module)
        class ImmediateThread:
            def __init__(self,target,**kwargs):self.target=target;self.args=kwargs.get('args',())
            def start(self):self.target(*self.args)
            def join(self):pass
        process=SimpleNamespace(pid=17,stdout=io.StringIO('output : 1 2\n'),stderr=io.StringIO('boot fixture\n'),
            poll=lambda:None,wait=lambda:0,returncode=0)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);reference=root/'fixture';reference.mkdir()
            (reference/'on.command.json').write_text(json.dumps(['strata.exe','--fixture']))
            with patch.object(module.ab,'T',str(root)),patch.dict(module.ab.REF,{'fixture':'fixture'}),\
                 patch.object(module.subprocess,'Popen',return_value=process),patch.object(module.threading,'Thread',ImmediateThread),\
                 patch.object(module.ab,'system_memory',side_effect=OSError('fixture sampling failure')):
                with self.assertRaisesRegex(RuntimeError,'memory telemetry'):
                    module.main(['run','fixture','never-executed.exe'])
            self.assertEqual((root/'run/stdout.txt').read_text(),'output : 1 2\n')
            self.assertIn('fixture sampling failure',(root/'run/memory-errors.json').read_text())
            self.assertEqual(len((root/'run/trace.tsv').read_text().splitlines()),1)

    def test_abba_keeps_raw_runs_but_never_prints_a_false_zero_memory_peak(self):
        class ImmediateThread:
            def __init__(self,target,args=(),**kwargs):self.target=target;self.args=args
            def start(self):self.target(*self.args)
            def join(self):pass
        launches=[]
        def fake_launch(args,**kwargs):
            launches.append((args,kwargs))
            return SimpleNamespace(pid=17,args=args,poll=lambda:None,returncode=0,
                communicate=lambda:('output : 1 2 3\n-> 10.00 tok/s\n','fixture stderr\n'))
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);reference=root/'reference';reference.mkdir()
            (reference/'on.command.json').write_text(json.dumps(['strata.exe','--fixture']))
            executable=root/'fixture.exe';executable.write_bytes(b'not executable; Popen is mocked')
            with patch.object(ab,'T',str(root)),patch.dict(ab.REF,{'fixture':'reference'}),\
                 patch.object(ab.subprocess,'Popen',side_effect=fake_launch),patch.object(ab.threading,'Thread',ImmediateThread),\
                 patch.object(ab,'system_memory',side_effect=OSError('fixture telemetry error')),patch('sys.stdout',new=io.StringIO()):
                ab.main(['measurement','A='+str(executable),'B='+str(executable),'--prompts','fixture'])
            self.assertEqual(len(launches),4)
            for _,kwargs in launches:self.assertEqual(kwargs['creationflags'],0x80|0x08000000)
            output=(root/'measurement/fixture-1-A.stdout').read_text()
            self.assertIn('memory unavailable',output);self.assertNotIn('memory peak',output)
            _,values=ab.parse_run(output);self.assertNotIn('sys_commit_gib',values)
            self.assertIn('memory=unavailable',(root/'measurement/summary.log').read_text())


if __name__=='__main__':unittest.main()
