"""Analytical checks for calibration and paired statistical inference."""
import numpy as np
import pandas as pd
import pytest
from bua_lel.utils.calibration import BinaryCalibrationAccumulator
from analysis.statistical_comparison import compare


def test_perfect_binary_calibration():
    accumulator=BinaryCalibrationAccumulator(5)
    accumulator.update([0,1,0,1],[0,1,0,1])
    result=accumulator.compute()
    assert result['ece']==0 and result['brier_score']==0


def test_identical_paired_predictions_have_zero_difference():
    frame=pd.DataFrame({'pid':['a','b','c','d'],'fold':[1,1,2,2],'y_true':[0,1,0,1],'proba_0':[0.8,0.2,0.9,0.1],'proba_1':[0.2,0.8,0.1,0.9]})
    result=compare(frame,frame,'ACC',n=10)
    assert result['difference']==0 and result['p_value']==1


def test_statistical_comparison_rejects_different_folds():
    frame=pd.DataFrame({'pid':['a','b'],'fold':[1,2],'Dice':[0.8,0.9]})
    other=frame.copy();other['fold']=[2,1]
    with pytest.raises(ValueError,match='fold'):
        compare(frame,other,'Dice',n=10)
