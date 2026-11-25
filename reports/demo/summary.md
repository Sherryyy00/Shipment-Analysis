# Results summary

- Input: `data/sample/demo_reviews_synthetic.csv`
- Reviews analysed: **897**
- Headline sentiment model: **VADER + courier lexicon**

## Couriers

Net sentiment is scored on the sentences about each courier. 'Whole review' scores the full review, which includes the shop and product.

| Courier | Reviews | Avg ★ | % negative | Net sentiment | Whole review |
|---|---:|---:|---:|---:|---:|
| Yodel | 117 | 2.57 | 45% | -2 | -5 |
| Evri | 234 | 2.60 | 44% | +2 | -2 |
| Royal Mail | 179 | 2.94 | 35% | +18 | +20 |
| DHL | 51 | 3.61 | 24% | +45 | +45 |
| Amazon Logistics | 83 | 3.42 | 25% | +49 | +39 |
| DPD | 197 | 3.77 | 19% | +56 | +55 |
| InPost | 36 | 3.61 | 11% | +67 | +67 |

## Top complaint aspects

- Delivery speed: 47% of negative reviews
- Customer service: 34% of negative reviews
- Driver behaviour: 33% of negative reviews

## Model agreement with star ratings

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| VADER + courier lexicon | 80% | 0.63 |
| VADER | 77% | 0.61 |
| TextBlob | 80% | 0.60 |

## Complaint themes

- (26%) late, gave chatbot, customer, customer gave, chatbot, gave, late customer, address
- (23%) arrived, thrown, arrived damaged, damaged thrown, fence, damaged, thrown fence, marked arrived
- (14%) tracking useless, useless, tracking, joke, ordering tracking, useless thats, ordering, useless tracking
- (13%) speak, speak person, impossible, impossible speak, person, lost tried, tried impossible, tried
- (13%) doorstep walked, dumped doorstep, knock, photo knock, photo, walked, walked photo, doorstep
- (11%) waiting refund, refund, refund lost, avoid, waiting, lost avoid, lost, joke

## Praise themes

- (31%) driver, right, right driver, driver friendly, friendly, smiled waited, driver smiled, waited door
- (21%) informed way, quick, quick kept, way, kept, informed, kept informed, brilliant
- (14%) stuck, hour slot, great tracking, tracking hour, hour, slot, slot stuck, tracking
- (13%) app, exactly, impressed, impressed turned, exactly app, turned exactly, turned, app thats
- (11%) condition, handed, perfect condition, polite, polite handed, perfect, door box, box perfect
- (10%) fine, vague, bit vague, bit, came fine, tracking bit, fine tracking, came
