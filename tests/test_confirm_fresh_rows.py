import pandas as pd
from scripts.confirm_fresh_rows import load_shards


def test_repeated_local_ids_are_unique_across_shards(tmp_path):
    for i in range(6):
        p = tmp_path / f'shard{i}'
        p.mkdir()
        pd.DataFrame({'pair_id':['p1','p2'], 'x':[i,i+1]}).to_csv(p/'baseline.csv',index=False)
    d = load_shards(tmp_path, 'baseline.csv')
    assert len(d)==12 and d.pair_id.is_unique
    assert d.iloc[2].pair_id=='shard1:p1'
