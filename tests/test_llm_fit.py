import json
import os
import subprocess
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import llm_fit as lf  # noqa: E402

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "llm_fit.py")


def args(**kw):
    base = dict(use="any", ctx=8192, min_speed=8.0)
    base.update(kw)
    return SimpleNamespace(**base)


def hw(ram=32, vram=None, gpu_bw=500, ram_bw=60, unified=False):
    h = lf.Hardware(ram_total_gb=ram, ram_bw_gbs=ram_bw, disk_free_gb=500, unified=unified)
    if unified:
        h.gpus = [lf.GPU("Apple M4 Pro", "Apple", ram, gpu_bw)]
    elif vram:
        h.gpus = [lf.GPU("Test GPU", "NVIDIA", vram, gpu_bw)]
    return h


def model(tag):
    return next(m for m in lf.DEFAULT_CATALOG if m["tag"] == tag)


class MemoryTests(unittest.TestCase):
    def test_weights_scale_with_bits(self):
        m = model("qwen3:8b")
        w8, _, _ = lf.memory_need(m, 8.5, 8192)
        w4, _, _ = lf.memory_need(m, 4.85, 8192)
        self.assertAlmostEqual(w8, 8.2 * 8.5 / 8, places=3)
        self.assertLess(w4, w8)

    def test_context_increases_memory(self):
        m = model("qwen3:14b")
        _, _, short = lf.memory_need(m, 4.85, 8192)
        _, _, long = lf.memory_need(m, 4.85, 65536)
        self.assertGreater(long, short)

    def test_fixed_size_models(self):
        w, _, _ = lf.memory_need(model("gpt-oss:20b"), 0, 8192)
        self.assertEqual(w, 13.8)


class PlacementTests(unittest.TestCase):
    def test_small_model_goes_on_gpu(self):
        m = model("qwen3:8b")
        w, _, need = lf.memory_need(m, 4.85, 8192)
        where, tps = lf.place(m, w, need, hw(vram=12))
        self.assertEqual(where, "GPU")
        self.assertGreater(tps, 30)

    def test_split_when_vram_too_small(self):
        m = model("qwen3:32b")
        w, _, need = lf.memory_need(m, 4.85, 8192)
        where, _ = lf.place(m, w, need, hw(vram=12, ram=64))
        self.assertTrue(where.startswith("Split"))

    def test_cpu_only(self):
        m = model("qwen3:4b")
        w, _, need = lf.memory_need(m, 4.85, 8192)
        where, _ = lf.place(m, w, need, hw(ram=16))
        self.assertEqual(where, "CPU")

    def test_does_not_fit(self):
        m = model("llama3.3:70b")
        w, _, need = lf.memory_need(m, 3.9, 8192)
        self.assertIsNone(lf.place(m, w, need, hw(ram=16)))

    def test_moe_faster_than_dense_of_same_size(self):
        h = hw(vram=24, gpu_bw=1000, ram=64)
        dense, moe = model("qwen3:32b"), model("qwen3:30b")
        wd, _, nd = lf.memory_need(dense, 4.85, 8192)
        wm, _, nm = lf.memory_need(moe, 4.85, 8192)
        self.assertGreater(lf.place(moe, wm, nm, h)[1], lf.place(dense, wd, nd, h)[1] * 3)

    def test_unified_memory_never_splits(self):
        m = model("llama3.3:70b")
        w, _, need = lf.memory_need(m, 4.85, 8192)
        self.assertIsNone(lf.place(m, w, need, hw(ram=36, unified=True, gpu_bw=273)))


class RankingTests(unittest.TestCase):
    def test_bigger_machine_gets_better_top_pick(self):
        small, _, _ = lf.evaluate(lf.DEFAULT_CATALOG, hw(ram=16), args())
        big, _, _ = lf.evaluate(lf.DEFAULT_CATALOG, hw(vram=24, gpu_bw=1000, ram=64), args())
        self.assertGreater(big[0]["quality"], small[0]["quality"])

    def test_use_coding_prefers_coding_models(self):
        fits, _, _ = lf.evaluate(lf.DEFAULT_CATALOG, hw(vram=12, ram=32), args(use="coding"))
        self.assertIn("coding", fits[0]["uses"])

    def test_min_speed_respected(self):
        fits, _, _ = lf.evaluate(lf.DEFAULT_CATALOG, hw(ram=32), args(min_speed=20))
        self.assertTrue(all(r["tok_s"] >= 20 for r in fits))

    def test_every_model_is_classified(self):
        fits, slow, nofit = lf.evaluate(lf.DEFAULT_CATALOG, hw(ram=16), args())
        self.assertEqual(len(fits) + len(slow) + len(nofit), len(lf.DEFAULT_CATALOG))

    def test_catalog_entries_are_valid(self):
        for m in lf.DEFAULT_CATALOG:
            self.assertTrue({"name", "tag", "params_b", "quality", "uses"} <= set(m), m["name"])
            if m.get("active_b"):
                self.assertLess(m["active_b"], m["params_b"])


class CliTests(unittest.TestCase):
    def run_cli(self, *extra):
        return subprocess.run([sys.executable, SCRIPT, *extra], capture_output=True, text=True, timeout=60)

    def test_json_output(self):
        p = self.run_cli("--json", "--vram", "8", "--gpu-name", "RTX 3060 Ti", "--ram", "32")
        self.assertEqual(p.returncode, 0, p.stderr)
        data = json.loads(p.stdout)
        self.assertIn("ranked", data)
        self.assertTrue(data["ranked"])

    def test_report_runs_on_real_hardware(self):
        p = self.run_cli("--all")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("Your system", p.stdout)


if __name__ == "__main__":
    unittest.main()
