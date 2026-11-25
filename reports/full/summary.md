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

- (41%) arrived, late, failed, problem, unfortunately, poor, arrive, damaged
- (16%) left, door, house, driver, leave, box, outside, bell
- (15%) lost, customer, refund, replacement, post, christmas, wait, money
- (14%) received, tracking, email, number, tracking number, december, received email, response
- (8%) address, wrong, wrong address, correct, delivering, correct address, refund, house
- (7%) good, product, fault, probably, price, communication, completely, products

## Praise themes

- (52%) customer, received, left, address, issue, email, post, paid
- (15%) arrived, excellent, quickly, packed, condition, packaged, helpful, free
- (10%) good, easy, good price, price, product, good quality, late, good customer
- (9%) great, product, products, price, quality, great price, quick, selection
- (9%) tracking, number, tracking number, received, email, information, provided, track
- (5%) fault, box, despite, expected, happy, better, perfect, attempt
