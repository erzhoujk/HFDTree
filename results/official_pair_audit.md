# Official pair-data audit

## Reproduced results

- Observed pairs: `{'AgentHarm': 208, 'AgentSafetyBench': 129, 'total_pairs': 337}`
- Embedded AgentHarm records exactly match source: `True`
- Embedded Agent-SafetyBench records exactly match source: `True`
- AgentHarm pairs with identical target functions: `171/208`
- Current strict Agent-SafetyBench pairing is maximum-cardinality: `True`

## Pairing-condition sensitivity

| Condition | Maximum one-to-one pairs |
|---|---:|
| Same risk + same environment + shared tool | 129 |
| Same environment + shared tool | 187 |
| Shared tool | 297 |
| Shared environment | 435 |
| Shared risk | 699 |

## Interpretation boundary

The 337 objects are official task/record pairs. They contain zero verified executed trajectory pairs and cannot be used as `c+`/`c-` until both outcomes are produced and scored from a compatible environment state. AgentHarm must remain evaluation-only because its official dataset card asks researchers not to train on its examples.
