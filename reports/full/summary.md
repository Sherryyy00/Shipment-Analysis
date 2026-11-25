# Results summary

- Input: `data/raw/trustpilot_reviews_2005.csv`
- Reviews analysed: **3,592**
- Headline sentiment model: **VADER + courier lexicon**

## Couriers

Net sentiment is scored on the sentences about each courier. 'Whole review' scores the full review, which includes the shop and product.

| Courier | Reviews | Avg ★ | % negative | Net sentiment | Whole review |
|---|---:|---:|---:|---:|---:|
| Yodel | 220 | 2.36 | 65% | -43 | -28 |
| FedEx | 172 | 2.48 | 59% | -30 | -8 |
| Evri | 623 | 2.55 | 56% | -26 | -10 |
| Parcelforce | 195 | 2.79 | 54% | -24 | +4 |
| InPost | 44 | 2.55 | 50% | -16 | -18 |
| DPD | 934 | 2.66 | 49% | -11 | -8 |
| DHL | 400 | 2.73 | 47% | -10 | +4 |
| UPS | 157 | 2.69 | 48% | -8 | +3 |
| Royal Mail | 1,075 | 2.79 | 46% | -7 | +4 |

## Top complaint aspects

- Delivery speed: 35% of negative reviews
- Customer service: 32% of negative reviews
- Tracking & updates: 24% of negative reviews

## Model agreement with star ratings

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| TextBlob | 52% | 0.48 |
| VADER + courier lexicon | 58% | 0.46 |
| VADER | 57% | 0.45 |

## Complaint themes

- (27%) lost, customer, refund, problem, sent, arrive, took, send
- (19%) received, tracking, email, number, saying, tracking number, december, received email
- (18%) left, door, house, driver, box, did, leave, outside
- (17%) arrived, late, unfortunately, arrived late, damaged, failed, christmas, product
- (11%) good, let, force, unfortunately, product, fault, communication, experience
- (8%) address, wrong, wrong address, correct, delivering, correct address, force, sent

## Praise themes

- (37%) customer, did, like, left, address, issue, box, arrive
- (16%) arrived, excellent, quickly, packed, condition, helpful, packaged, hours
- (15%) tracking, number, tracking number, received, email, information, sent, provided
- (14%) great, product, products, price, let, great price, quick, quality
- (12%) good, let, easy, good price, price, good quality, late, quality
- (6%) fault, box, expected, happy, despite, perfect, better, attempt
