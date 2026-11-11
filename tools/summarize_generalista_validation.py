"""Consolida as baterias congeladas e confere a promoção da política oficial."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('camper', 'strafe', 'dijkstra', 'astar', 'markov')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows):
    def group(items):
        n = len(items)
        wins = sum(r['winner'] == 0 for r in items)
        draws = sum(r['winner'] == -1 for r in items)
        return dict(matches=n, wins=wins, draws=draws, losses=n-wins-draws,
                    win_rate=wins/n)
    return dict(total=group(rows), by_opponent={n: group([r for r in rows if r['opponent'] == n]) for n in NAMES})


def main(artifacts):
    def read(name):
        return json.loads((artifacts / name).read_text(encoding='utf-8'))
    historical = [r for name in ('threshold_markov_boost6_boost10_quick8_quick10.json', 'threshold_others_quick10.json')
                  for r in read(name) if r['variant'] == 'quick10']
    fresh = read('threshold_fresh40_quick10.json')
    bots = read('builtin_quick10.json')
    bot_base = read('builtin_current.json')
    previous = json.loads((ROOT/'MeuAlgo/experiments/markov_return_results.json').read_text(encoding='utf-8'))['matches']
    key = lambda r: (r['opponent'], r['seed'], r['swapped'])
    assert {key(r) for r in historical} == {key(r) for r in previous}
    assert not {r['seed'] for r in historical} & {r['seed'] for r in fresh}
    assert {key(r) for r in bots} == {key(r) for r in bot_base}
    for rows, count in ((historical, 70), (fresh, 40), (bots, 20), (bot_base, 20)):
        assert len(rows) == len({key(r) for r in rows}) == count*5
        assert all(v['matches'] == count for v in summarize(rows)['by_opponent'].values())
    for rows in (historical, fresh):
        assert all(v['win_rate'] >= .75 for v in summarize(rows)['by_opponent'].values())
    promotion = read('before_promotion/manifest.json')
    for path, hashes in promotion.items():
        assert sha(ROOT/path) == hashes['after']
        assert sha(artifacts/'before_promotion'/path) == hashes['before']
    assert sha(ROOT/'MeuAlgo/algo1.py') == sha(artifacts/'quick10.py') == sha(ROOT/'MeuAlgo/experiments/generalista_candidate.py')
    manifests = ('threshold_markov_boost6_boost10_quick8_quick10.manifest.json',
                 'threshold_others_quick10.manifest.json', 'threshold_fresh40_quick10.manifest.json',
                 'builtin_quick10.manifest.json', 'builtin_current.manifest.json')
    for name in manifests:
        for path, expected in read(name)['inputs'].items():
            target = Path(path)
            if target == ROOT/'MeuAlgo/algo1.py':
                assert expected == promotion['MeuAlgo/algo1.py']['before']
            else:
                assert sha(target) == expected, path
    equivalence = [json.loads(line) for line in (artifacts/'equivalence_final.log').read_text().splitlines() if line.startswith('{')]
    assert equivalence[-1]['matches'] == 12 and equivalence[-1]['all_actions_identical']
    keep = ('opponent', 'seed', 'bot_seed', 'swapped', 'winner', 'lives', 'seconds', 'shots', 'hits', 'heat')
    compact = lambda rows: [{k:r[k] for k in keep if k in r} for r in sorted(rows, key=key)]
    report = dict(date='2026-11-10', status='promoted_to_main_and_both_contest_slots',
        target_minimum_win_rate=.75, target_met_in_both_specialist_samples=True,
        protocol=dict(specialists='Frozen trained specialist checkpoints, not built-in bots',
                      duration='Natural engine termination, maximum 120 seconds', both_spawn_sides=True,
                      historical='35 maps per specialist; used during development',
                      independent='20 new maps, frozen policy, no adjustments after observing results',
                      independent_map_seed_generator='random.Random(2026111057).sample(range(70000000,80000000),20)',
                      bots='10 maps (1120000..1120009), both sides, bot seed = map seed + 1',
                      limitation='Observed sample win rates; no guarantee for every future map'),
        changes=['Immediate-hit defense tie break', 'Deadline-aware safe return and teleport cache invalidation',
                 'Nearby safe LIFE/SPEED/QUICKSHOT collection', 'Immediate tactical use of temporary speed or fire-rate advantage'],
        weights_unchanged=True, opponent_name_or_seed_specific_policy=False,
        historical=summarize(historical), independent=summarize(fresh),
        previous_development=summarize(previous), builtin_before=summarize(bot_base), builtin_after=summarize(bots),
        validation=dict(policy_tests=40, project_checks=23, project_specialist_regressions=32,
                        contest_equivalence=equivalence[-1]), promotion=promotion,
        provenance=dict(artifacts=str(artifacts), input_manifests={n:read(n) for n in manifests},
                        raw_files={n:sha(artifacts/n) for n in ('threshold_markov_boost6_boost10_quick8_quick10.json',
                            'threshold_others_quick10.json','threshold_fresh40_quick10.json','builtin_current.json','builtin_quick10.json')}),
        matches=dict(historical=compact(historical), independent=compact(fresh), builtin_before=compact(bot_base), builtin_after=compact(bots)))
    output = ROOT/'MeuAlgo/experiments/generalista_75_each_results.json'
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps({k:report[k]['total'] for k in ('historical','independent','builtin_after')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('artifacts', type=Path)
    main(parser.parse_args().artifacts)
