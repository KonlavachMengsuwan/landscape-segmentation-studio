import numpy as np
import pytest
from backend.overlap import suppress_overlaps,validate_overlap

def result(indices,score):
 mask=np.zeros((3,4),bool);mask.flat[indices]=True
 return dict(mask=mask,score=score)

def test_nested_smaller_metric_and_iou_are_distinct_and_reversible():
 results=[result([0,1,2,3],.9),result([0,1],.8),result([9,10],.7)]
 before=[r['mask'].copy() for r in results]
 assert suppress_overlaps(results,dict(remove_overlap=True,overlap_threshold=80))[1]['suppressed_by_index']==1
 assert suppress_overlaps(results,dict(remove_overlap=True,overlap_threshold=80,overlap_metric='iou'))==[None]*3
 assert all(np.array_equal(r['mask'],m) for r,m in zip(results,before))
 assert suppress_overlaps(results,{})==[None]*3

def test_score_order_ties_endpoints_and_disjoint_masks():
 results=[result([0,1],.5),result([0,1],.9),result([0,1],.9),result([2,3],.9)]
 records=suppress_overlaps(results,dict(remove_overlap=True,overlap_threshold=100))
 assert records[0]['suppressed_by_index']==2 and records[2]['suppressed_by_index']==2
 assert records[1] is records[3] is None
 assert suppress_overlaps([result([0],1),result([1],0)],dict(remove_overlap=True,overlap_threshold=0))==[None,None]

@pytest.mark.parametrize('settings',[dict(remove_overlap=1),dict(overlap_threshold=-1),dict(overlap_threshold=101),dict(overlap_threshold=float('nan')),dict(overlap_metric='box')])
def test_invalid(settings):
 with pytest.raises(ValueError):validate_overlap(settings)
