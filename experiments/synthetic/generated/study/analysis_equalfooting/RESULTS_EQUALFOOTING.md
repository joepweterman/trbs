# A4 equal-footing study: results

Exploratory supplement to the locked analysis. It revises no locked verdict.
Rows read: 5,040 synthetic and 54 packaged.

## Predictions

| prediction | verdict | evidence |
| --- | --- | --- |
| P1 | FAILS | IZZ median closed by capping -0.0031762487841149323, Beerwiser 0.0 |
| P2 | FAILS | grid reached 1.0 everywhere: True, best fraction reaching 0.01 at k>=6: 0.9666666666666667 |
| P3 | FAILS | median share of runtime that is not evaluation, k>=9: 0.0 |
| P4 | FAILS | mdbh/slsqp evaluations to target: {'1': 38.0, '0.1': 79.89204545454545, '0.01': 122.15} |

## Constraint decomposition on the packaged cases

| case_name | k | scenario | reference | gap_face | gap_capped | closed_by_capping | spend_face | spend_capped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Beerwiser | 2 | Base case | 65.7116 | 0 | 0 | 0 | 1 | 1 |
| Beerwiser | 2 | Optimistic | 65.9549 | 0 | 0 | 0 | 1 | 1 |
| Beerwiser | 2 | Pessimistic | 83.3333 | 0 | 0 | 0 | 1 | 1 |
| IZZ | 9 | Base case | 69.1905 | 0.116936 | 0.458314 | -0.341378 | 1 | 1 |
| IZZ | 9 | Optimistic | 69.1907 | 0.062783 | 0.0659593 | -0.00317625 | 1 | 1 |
| IZZ | 9 | Pessimistic | 61.6276 | 1.06394 | 0.748387 | 0.315551 | 1 | 1 |
| Refugee | 5 | Base case | 96.8689 | 0.0604269 | 0.413339 | -0.352912 | 1 | 1 |
| Refugee | 5 | Labour optimistic | 99.9267 | 0.00765724 | 0.0528606 | -0.0452034 | 1 | 1 |
| Refugee | 5 | Labour pessimistic | 76.9931 | 0.188542 | 0.188542 | 0 | 1 | 1 |

## Overhead: the share of wall-clock that is not objective evaluation

### Packaged

| method | k | median_share | n |
| --- | --- | --- | --- |
| basin_hopping | 2 | 0.011414 | 3 |
| basin_hopping | 5 | 0.00197292 | 3 |
| basin_hopping | 9 | 0.00293766 | 3 |
| genetic_algorithm | 2 | 0 | 3 |
| genetic_algorithm | 5 | 0 | 3 |
| genetic_algorithm | 9 | 0 | 3 |
| grid | 2 | 0.485715 | 3 |
| grid | 5 | 0 | 3 |
| grid | 9 | 0 | 3 |
| grid_capped | 2 | 0 | 3 |
| grid_capped | 5 | 0 | 3 |
| grid_capped | 9 | 0 | 3 |
| mdbh | 2 | 0 | 3 |
| mdbh | 5 | 0 | 3 |
| mdbh | 9 | 0 | 3 |
| slsqp | 2 | 0 | 3 |
| slsqp | 5 | 0 | 3 |
| slsqp | 9 | 0 | 3 |

### Synthetic

| method | k | median_share | n |
| --- | --- | --- | --- |
| basin_hopping | 2 | 0.170274 | 180 |
| basin_hopping | 3 | 0.177719 | 180 |
| basin_hopping | 4 | 0.134134 | 180 |
| basin_hopping | 6 | 0.0988178 | 180 |
| genetic_algorithm | 2 | 0 | 180 |
| genetic_algorithm | 3 | 0 | 180 |
| genetic_algorithm | 4 | 0 | 180 |
| genetic_algorithm | 6 | 0 | 180 |
| grid | 2 | 0 | 240 |
| grid | 3 | 0 | 240 |
| grid | 4 | 0 | 240 |
| grid | 6 | 0 | 240 |
| grid_capped | 2 | 0 | 240 |
| grid_capped | 3 | 0 | 240 |
| grid_capped | 4 | 0 | 240 |
| grid_capped | 6 | 0 | 240 |
| mdbh | 2 | 0 | 180 |
| mdbh | 3 | 0 | 180 |
| mdbh | 4 | 0 | 180 |
| mdbh | 6 | 0 | 180 |
| slsqp | 2 | 0.114387 | 240 |
| slsqp | 3 | 0.0526024 | 240 |
| slsqp | 4 | 0.0478467 | 240 |
| slsqp | 6 | 0.0247176 | 240 |

## Data profiles at the frozen targets

| tau | method | k | reached | n | fraction | wilson_low | wilson_high | median_evals |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | basin_hopping | 2 | 60 | 60 | 1 | 0.939828 | 1 | 4 |
| 1 | basin_hopping | 3 | 60 | 60 | 1 | 0.939828 | 1 | 5 |
| 1 | basin_hopping | 4 | 60 | 60 | 1 | 0.939828 | 1 | 6 |
| 1 | basin_hopping | 6 | 60 | 60 | 1 | 0.939828 | 1 | 8 |
| 1 | genetic_algorithm | 2 | 60 | 60 | 1 | 0.939828 | 1 | 164 |
| 1 | genetic_algorithm | 3 | 60 | 60 | 1 | 0.939828 | 1 | 230.5 |
| 1 | genetic_algorithm | 4 | 57 | 60 | 0.95 | 0.862995 | 0.98285 | 663 |
| 1 | genetic_algorithm | 6 | 51 | 60 | 0.85 | 0.738854 | 0.919026 | 1386 |
| 1 | grid | 2 | 60 | 60 | 1 | 0.939828 | 1 | 2 |
| 1 | grid | 3 | 60 | 60 | 1 | 0.939828 | 1 | 1 |
| 1 | grid | 4 | 60 | 60 | 1 | 0.939828 | 1 | 2 |
| 1 | grid | 6 | 60 | 60 | 1 | 0.939828 | 1 | 3.5 |
| 1 | grid_capped | 2 | 60 | 60 | 1 | 0.939828 | 1 | 30658 |
| 1 | grid_capped | 3 | 60 | 60 | 1 | 0.939828 | 1 | 51 |
| 1 | grid_capped | 4 | 60 | 60 | 1 | 0.939828 | 1 | 337.5 |
| 1 | grid_capped | 6 | 60 | 60 | 1 | 0.939828 | 1 | 286 |
| 1 | mdbh | 2 | 60 | 60 | 1 | 0.939828 | 1 | 44.5 |
| 1 | mdbh | 3 | 60 | 60 | 1 | 0.939828 | 1 | 141 |
| 1 | mdbh | 4 | 60 | 60 | 1 | 0.939828 | 1 | 183.5 |
| 1 | mdbh | 6 | 60 | 60 | 1 | 0.939828 | 1 | 373 |
| 1 | slsqp | 2 | 60 | 60 | 1 | 0.939828 | 1 | 4 |
| 1 | slsqp | 3 | 60 | 60 | 1 | 0.939828 | 1 | 5 |
| 1 | slsqp | 4 | 60 | 60 | 1 | 0.939828 | 1 | 6 |
| 1 | slsqp | 6 | 60 | 60 | 1 | 0.939828 | 1 | 8 |
| 0.1 | basin_hopping | 2 | 60 | 60 | 1 | 0.939828 | 1 | 4 |
| 0.1 | basin_hopping | 3 | 60 | 60 | 1 | 0.939828 | 1 | 5 |
| 0.1 | basin_hopping | 4 | 60 | 60 | 1 | 0.939828 | 1 | 6 |
| 0.1 | basin_hopping | 6 | 60 | 60 | 1 | 0.939828 | 1 | 8 |
| 0.1 | genetic_algorithm | 2 | 60 | 60 | 1 | 0.939828 | 1 | 460 |
| 0.1 | genetic_algorithm | 3 | 60 | 60 | 1 | 0.939828 | 1 | 1008 |
| 0.1 | genetic_algorithm | 4 | 39 | 60 | 0.65 | 0.523626 | 0.758322 | 1201 |
| 0.1 | genetic_algorithm | 6 | 18 | 60 | 0.3 | 0.198982 | 0.425087 | 1221 |
| 0.1 | grid | 2 | 60 | 60 | 1 | 0.939828 | 1 | 2 |
| 0.1 | grid | 3 | 60 | 60 | 1 | 0.939828 | 1 | 2 |
| 0.1 | grid | 4 | 60 | 60 | 1 | 0.939828 | 1 | 2 |
| 0.1 | grid | 6 | 60 | 60 | 1 | 0.939828 | 1 | 6 |
| 0.1 | grid_capped | 2 | 60 | 60 | 1 | 0.939828 | 1 | 31571 |
| 0.1 | grid_capped | 3 | 60 | 60 | 1 | 0.939828 | 1 | 1220.5 |
| 0.1 | grid_capped | 4 | 60 | 60 | 1 | 0.939828 | 1 | 3276 |
| 0.1 | grid_capped | 6 | 60 | 60 | 1 | 0.939828 | 1 | 1987 |
| 0.1 | mdbh | 2 | 60 | 60 | 1 | 0.939828 | 1 | 143.5 |
| 0.1 | mdbh | 3 | 60 | 60 | 1 | 0.939828 | 1 | 334 |
| 0.1 | mdbh | 4 | 60 | 60 | 1 | 0.939828 | 1 | 834 |
| 0.1 | mdbh | 6 | 60 | 60 | 1 | 0.939828 | 1 | 1267.5 |
| 0.1 | slsqp | 2 | 60 | 60 | 1 | 0.939828 | 1 | 4 |
| 0.1 | slsqp | 3 | 60 | 60 | 1 | 0.939828 | 1 | 5 |
| 0.1 | slsqp | 4 | 60 | 60 | 1 | 0.939828 | 1 | 6 |
| 0.1 | slsqp | 6 | 60 | 60 | 1 | 0.939828 | 1 | 8 |
| 0.01 | basin_hopping | 2 | 60 | 60 | 1 | 0.939828 | 1 | 4 |
| 0.01 | basin_hopping | 3 | 60 | 60 | 1 | 0.939828 | 1 | 5 |
| 0.01 | basin_hopping | 4 | 60 | 60 | 1 | 0.939828 | 1 | 6 |
| 0.01 | basin_hopping | 6 | 60 | 60 | 1 | 0.939828 | 1 | 8 |
| 0.01 | genetic_algorithm | 2 | 60 | 60 | 1 | 0.939828 | 1 | 564 |
| 0.01 | genetic_algorithm | 3 | 60 | 60 | 1 | 0.939828 | 1 | 1103 |
| 0.01 | genetic_algorithm | 4 | 37 | 60 | 0.616667 | 0.490176 | 0.729117 | 1949 |
| 0.01 | genetic_algorithm | 6 | 12 | 60 | 0.2 | 0.118285 | 0.317818 | 1505 |
| 0.01 | grid | 2 | 60 | 60 | 1 | 0.939828 | 1 | 2 |
| 0.01 | grid | 3 | 60 | 60 | 1 | 0.939828 | 1 | 2 |
| 0.01 | grid | 4 | 60 | 60 | 1 | 0.939828 | 1 | 3 |
| 0.01 | grid | 6 | 58 | 60 | 0.966667 | 0.886362 | 0.990811 | 6 |
| 0.01 | grid_capped | 2 | 60 | 60 | 1 | 0.939828 | 1 | 31625 |
| 0.01 | grid_capped | 3 | 60 | 60 | 1 | 0.939828 | 1 | 2625 |
| 0.01 | grid_capped | 4 | 60 | 60 | 1 | 0.939828 | 1 | 3276 |
| 0.01 | grid_capped | 6 | 47 | 60 | 0.783333 | 0.6638 | 0.86877 | 3003 |
| 0.01 | mdbh | 2 | 60 | 60 | 1 | 0.939828 | 1 | 240.5 |
| 0.01 | mdbh | 3 | 60 | 60 | 1 | 0.939828 | 1 | 604 |
| 0.01 | mdbh | 4 | 60 | 60 | 1 | 0.939828 | 1 | 1368.5 |
| 0.01 | mdbh | 6 | 58 | 60 | 0.966667 | 0.886362 | 0.990811 | 1756 |
| 0.01 | slsqp | 2 | 60 | 60 | 1 | 0.939828 | 1 | 4 |
| 0.01 | slsqp | 3 | 60 | 60 | 1 | 0.939828 | 1 | 5 |
| 0.01 | slsqp | 4 | 60 | 60 | 1 | 0.939828 | 1 | 6 |
| 0.01 | slsqp | 6 | 60 | 60 | 1 | 0.939828 | 1 | 8 |

## Paired cost to target against SLSQP

| tau | method | n_pairs | median_ratio | statistic | p_raw | p_holm |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | basin_hopping | 240 | 1 | 0 | nan | 0 |
| 1 | genetic_algorithm | 228 | 82.5 | 45 | 6.68748e-39 | 1.3375e-38 |
| 1 | grid | 240 | 0.5 | 3813 | 7.16001e-21 | 7.16001e-21 |
| 1 | grid_capped | 240 | 339.9 | 0 | 3.96038e-41 | 1.58415e-40 |
| 1 | mdbh | 240 | 38 | 0 | 3.97474e-41 | 1.58415e-40 |
| 0.1 | basin_hopping | 240 | 1 | 0 | nan | 0 |
| 0.1 | genetic_algorithm | 177 | 151.4 | 0 | 8.47293e-31 | 1.69459e-30 |
| 0.1 | grid | 240 | 0.5 | 11236 | 0.00263911 | 0.00263911 |
| 0.1 | grid_capped | 240 | 546 | 0 | 3.84571e-41 | 1.53828e-40 |
| 0.1 | mdbh | 240 | 79.892 | 0 | 3.99141e-41 | 1.53828e-40 |
| 0.01 | basin_hopping | 240 | 1 | 0 | nan | 0 |
| 0.01 | genetic_algorithm | 169 | 172 | 0 | 1.73895e-29 | 3.4779e-29 |
| 0.01 | grid | 238 | 0.5 | 12803 | 0.217446 | 0.217446 |
| 0.01 | grid_capped | 227 | 1001 | 0 | 5.01807e-39 | 1.50542e-38 |
| 0.01 | mdbh | 238 | 122.15 | 0 | 8.49579e-41 | 3.39832e-40 |
