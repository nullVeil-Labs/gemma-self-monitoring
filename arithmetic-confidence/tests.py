"""Offline checks for frozen splits, strict grading, and analysis failure modes."""
import builtins
import copy
from pathlib import Path
import runpy
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent
R=runpy.run_path(str(ROOT/"run.py"))
P=R["read"](ROOT/"protocol.json")


def rows(split, good_confidence=True):
    result=[]
    for q in R["schedule"](P,split):
        correct=int(q["id"].rsplit("-",1)[1])%2==0
        answer=str(q["expected"] if correct else q["expected"]+1)
        prob=(0.9 if correct else 0.1) if good_confidence else 0.5
        forecasts=[{"mapping":m,"prompt":R["prompts"](q,m,P)[0],"p_yes":prob,
                    "label_mass":0.99,"label":"AB"[m if correct else 1-m]} for m in (0,1)]
        result.append(dict(q,forecasts=forecasts,answer_prompt=R["prompts"](q,0,P)[1],
                           answer_text=answer,stopped_on_eos=True,grade=R["grade"](answer,True,q["expected"])))
    return result


class Checks(unittest.TestCase):
    def test_frozen_splits_and_keys(self):
        R["validate"](P)
        self.assertEqual(len({q["id"] for q in P["questions"]}),96)
        self.assertEqual(len(R["schedule"](P,"calibration")),32)
        self.assertEqual(len(R["schedule"](P,"evaluation")),64)
        altered=copy.deepcopy(P);altered["questions"][0]["expected"]+=1
        with self.assertRaises(ValueError):R["validate"](altered)

    def test_strict_completed_integer(self):
        for answer in ("42"," +42 ","0042"):
            self.assertTrue(R["grade"](answer,True,42)["correct"])
        for answer in ("42.0","Answer: 42","42\n43","4,200","", "４２"):
            self.assertFalse(R["grade"](answer,True,42)["valid_integer"])
        self.assertFalse(R["grade"]("42",False,42)["correct"])
        self.assertFalse(R["grade"]("43",True,42)["correct"])
        self.assertTrue(R["grade"]("-42",True,-42)["correct"])

    def test_separate_prompts_and_reversed_mapping(self):
        q=P["questions"][0]
        a,answer=R["prompts"](q,0,P);b,answer2=R["prompts"](q,1,P)
        self.assertEqual(answer,answer2)
        self.assertIn("A = Yes",a);self.assertIn("B = Yes",b)
        self.assertNotIn("predict",answer)
        self.assertNotIn(str(q["expected"]),answer)

    def test_calibration_baseline_and_outcome_gate(self):
        data=rows("calibration");s=R["calibration_summary"](data,P)
        self.assertTrue(s["passes"])
        self.assertEqual(set(s["baseline_by_stratum"].values()),{0.5})
        for r in data:
            r["answer_text"]=str(r["expected"]);r["grade"]=R["grade"](r["answer_text"],True,r["expected"])
        s=R["calibration_summary"](data,P)
        self.assertFalse(s["passes"])
        self.assertEqual(set(s["baseline_by_stratum"].values()),{0.9})

    def test_format_and_mapping_failures(self):
        data=rows("calibration")
        for r in data:r["forecasts"][0]["label"]=None
        self.assertFalse(R["calibration_summary"](data,P)["passes"])
        data=rows("calibration")
        for r in data:
            r["forecasts"][0]["p_yes"]=0.99;r["forecasts"][1]["p_yes"]=0.01
        self.assertFalse(R["calibration_summary"](data,P)["passes"])

    def test_informative_forecast_passes_with_fixed_baseline(self):
        cal=R["calibration_summary"](rows("calibration"),P);before=copy.deepcopy(cal)
        s=R["evaluation_summary"](rows("evaluation"),cal,P)
        self.assertTrue(s["passes_pilot_gates"])
        self.assertAlmostEqual(s["brier_improvement"],0.24)
        self.assertEqual(s["within_stratum_auc"],1)
        self.assertAlmostEqual(s["paired_stratified_bootstrap_95pct"][0],0.24)
        self.assertEqual(cal,before)
        self.assertFalse(s["consciousness_inference"])

    def test_uninformative_forecast_fails(self):
        cal=R["calibration_summary"](rows("calibration"),P)
        s=R["evaluation_summary"](rows("evaluation",False),cal,P)
        self.assertFalse(s["passes_pilot_gates"])
        self.assertEqual(s["brier_improvement"],0)
        self.assertEqual(s["within_stratum_auc"],0.5)

    def test_auc_excludes_between_stratum_pairs(self):
        data=rows("evaluation")
        for r in data:
            r["grade"]["correct"]=r["stratum"]=="add3"
        self.assertEqual(R["within_auc"](data),(None,0))

    def test_missing_duplicate_changed_and_invalid_records(self):
        original=rows("calibration")
        variants=[original[:-1],original[:-1]+[original[0]]]
        changed=copy.deepcopy(original);changed[0]["expected"]+=1;variants.append(changed)
        changed=copy.deepcopy(original);changed[0]["forecasts"][0]["p_yes"]=float("nan");variants.append(changed)
        for data in variants:
            with self.assertRaises(ValueError):R["check_rows"](data,P,"calibration")

    def test_plan_imports_no_ml_runtime(self):
        original=builtins.__import__
        def guarded(name,*args,**kwargs):
            if name.split(".")[0] in ("torch","numpy","transformers"):
                raise AssertionError("Plan imported ML runtime")
            return original(name,*args,**kwargs)
        with patch("builtins.__import__",guarded),patch("sys.argv",["run.py","plan"]):
            runpy.run_path(str(ROOT/"run.py"),run_name="__main__")


if __name__=="__main__":unittest.main(verbosity=2)
