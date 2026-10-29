import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from tools.igc.provenance import verify_sources


class CapturedSources(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.original=b'captured build bytes'
        self.source=self.root/'reused-build.exe';self.source.write_bytes(self.original)
        self.sha=hashlib.sha256(self.original).hexdigest()
        st=self.source.stat()
        self.manifest=self.root/'sources.json'
        self.manifest.write_text(json.dumps(dict(verified=True,files=[dict(path=str(self.source),
            bytes=st.st_size,mtime_ns=st.st_mtime_ns,sha256=self.sha)])))

    def test_unchanged_source_and_changed_source_are_distinct(self):
        self.assertEqual(verify_sources(self.manifest)[0]['verification'],'captured-content-unchanged-stat')
        self.source.write_bytes(b'a new build')
        with self.assertRaises(ValueError):verify_sources(self.manifest)

    def test_historical_source_requires_exact_frozen_content_not_a_cache_name(self):
        cache=self.root/'frozen-inputs';cache.mkdir()
        frozen=cache/(self.sha+'.bin');frozen.write_bytes(self.original)
        self.source.write_bytes(b'a new build')
        record=verify_sources(self.manifest)[0]
        self.assertEqual(record['verification'],'frozen-content-sha256')
        self.assertEqual(record['resolved'],str(frozen))
        frozen.write_bytes(b'wrong bytes under a plausible name')
        with self.assertRaises(ValueError):verify_sources(self.manifest)

    def test_relocated_timestamp_can_be_reverified_by_content(self):
        data=json.loads(self.manifest.read_text());data['files'][0]['mtime_ns']=0
        self.manifest.write_text(json.dumps(data))
        self.assertEqual(verify_sources(self.manifest)[0]['verification'],'current-content-sha256')


if __name__=='__main__':
    unittest.main()
