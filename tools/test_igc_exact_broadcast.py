import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest


def integer(value):
    digits=bytes(map(int,reversed(str(abs(value)))))
    return bytes([len(digits),int(value<0)])+digits+bytes(62-len(digits))


@unittest.skipUnless(os.environ.get('IGC_EXACT_BROADCAST'),'native exact broadcast not selected')
class ExactBroadcast(unittest.TestCase):
    def test_graph_reuse_broadcast_and_error_never_create_a_successful_zero(self):
        exe=Path(os.environ['IGC_EXACT_BROADCAST'])
        if not exe.is_file():raise RuntimeError('broadcast executable missing')
        with tempfile.TemporaryDirectory(prefix='igc-broadcast-') as name:
            root=Path(name);source=root/'request.bin'
            for case,second_domain,second_op,second_answer,second_status in [('valid',1,2,96,0),('unsupported',2,1,None,2)]:
                source.write_bytes(struct.pack('<IIIIQQ64s64s',1,1,3,0,7,31,integer(93),integer(-3))+
                    struct.pack('<IIIIQQ64s64s',1,second_domain,second_op,0,8,77,integer(53),integer(-43)))
                for count in (1,64):
                    prefix=root/f'{case}-{count}'
                    result=subprocess.run([str(exe),str(source),str(prefix),str(count)],capture_output=True,creationflags=0x08000000,timeout=30)
                    self.assertEqual(result.returncode,0,result.stderr.decode(errors='replace'))
                    for index in range(3):
                        raw=Path(str(prefix)+f'.{index}.bin').read_bytes();self.assertEqual(len(raw),count*88)
                        expected=struct.pack('<IIQQ64s',1,second_status if index==1 else 0,8 if index==1 else 7,
                            77 if index==1 else 31,bytes(64) if index==1 and second_answer is None else integer(second_answer if index==1 else -279))
                        self.assertEqual(raw,expected*count)


if __name__=='__main__':unittest.main()
