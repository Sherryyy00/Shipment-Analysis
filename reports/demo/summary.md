# Results summary

- Input: `data/sample/demo_reviews_synthetic.csv`
- Reviews analysed: **897**
- Headline sentiment model: **VADER + courier lexicon**

## Couriers

| Courier | Reviews | Avg ★ | % negative | Net sentiment |
|---|---:|---:|---:|---:|
| Yodel | 117 | 2.57 | 50% | -5 |
| Evri | 234 | 2.60 | 47% | -2 |
| Royal Mail | 179 | 2.94 | 37% | +20 |
| Amazon Logistics | 83 | 3.42 | 30% | +39 |
| DHL | 51 | 3.61 | 25% | +45 |
| DPD | 197 | 3.77 | 20% | +55 |
| InPost | 36 | 3.61 | 14% | +67 |

## Top complaint aspects

- Delivery speed: 44% of negative reviews
- Driver behaviour: 34% of negative reviews
- Customer service: 31% of negative reviews

## Model agreement with star ratings

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| VADER + courier lexicon | 80% | 0.63 |
| VADER | 77% | 0.61 |
| TextBlob | 80% | 0.60 |

## Complaint themes

- (28%) dumped doorstep, dumped, doorstep walked, doorstep, photo knock, photo, walked photo, knock
- (19%) address, wrong address, wrong, address neighbour, bring, bring round, round, neighbour
- (18%) broken, box crushed, broken joke, crushed, contents, joke, crushed contents, contents broken
- (13%) speak, speak person, impossible, impossible speak, tried impossible, person, lost tried, tried
- (12%) arrived, arrived damaged, damaged, damaged thrown, thrown, thrown fence, fence, marked
- (9%) left outside, soaked, driver left, driver, box soaked, rain, rain box, outside rain

## Praise themes

- (37%) fast reliable, fast, reliable, reliable complaints, complaints, friendly, driver friendly, right driver
- (13%) door box, condition, handed door, handed, box perfect, polite handed, perfect condition, perfect
- (13%) driver smiled, smiled waited, waited door, lovely driver, smiled, lovely, waited, driver
- (12%) quick kept, quick, way, informed, kept, informed way, kept informed, mum quick
- (12%) stuck, tracking hour, great tracking, hour slot, slot, slot stuck, hour, tracking
- (12%) app, exactly, exactly app, turned exactly, impressed, impressed turned, turned, app thats
