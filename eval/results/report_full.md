# Evaluation report

Generated 2026-10-13T10:43:54  
Embeddings: `openai:nomic-embed-text` | LLM: `qwen3:4b-instruct-2507-q4_K_M` | chunk_size=500, overlap=80, top_k=6, hybrid=True, min_relevance=0.55

## Summary

```json
{
  "questions": 19,
  "retrieval": {
    "source_recall": 1.0,
    "mrr": 1.0,
    "context_recall": 1.0,
    "context_recall_multi_evidence": 1.0
  },
  "answers": {
    "pass_rate": 0.895,
    "answerable_pass_rate": 0.857,
    "fact_coverage": 0.857,
    "citation_precision": 0.929,
    "false_abstentions": 1,
    "abstention_accuracy_unanswerable": 1.0,
    "hallucinated_answers_to_unanswerable": 0,
    "median_latency_s": 4.77
  }
}
```

## Per question

| id | category | passed | status / reason | max sim | context recall | fact coverage | citation precision | latency s |
|---|---|---|---|---|---|---|---|---|
| A1 | answerable | yes | answered | 0.831 | 1.0 | 1.0 | 1.0 | 9.05 |
| A2 | answerable | yes | answered | 0.794 | 1.0 | 1.0 | 1.0 | 2.95 |
| A3 | answerable | yes | answered | 0.822 | 1.0 | 1.0 | 1.0 | 2.97 |
| A4 | answerable | yes | answered | 0.791 | 1.0 | 1.0 | 1.0 | 8.3 |
| A5 | answerable | yes | answered | 0.882 | 1.0 | 1.0 | 1.0 | 3.81 |
| A6 | answerable | yes | answered | 0.841 | 1.0 | 1.0 | 1.0 | 9.09 |
| A7 | answerable | yes | answered | 0.781 | 1.0 | 1.0 | 1.0 | 4.77 |
| A8 | answerable | **no** | unavailable / not_in_documents | 0.712 | 1.0 | 0.0 | 0.0 | 1.63 |
| A9 | answerable | **no** | answered | 0.764 | 1.0 | 0.0 | 1.0 | 12.2 |
| M1 | multi_evidence | yes | answered | 0.746 | 1.0 | 1.0 | 1.0 | 14.65 |
| M2 | multi_evidence | yes | answered | 0.723 | 1.0 | 1.0 | 1.0 | 6.11 |
| M3 | multi_evidence | yes | answered | 0.802 | 1.0 | 1.0 | 1.0 | 10.66 |
| M4 | multi_evidence | yes | answered | 0.741 | 1.0 | 1.0 | 1.0 | 6.46 |
| M5 | multi_evidence | yes | answered | 0.765 | 1.0 | 1.0 | 1.0 | 9.58 |
| U1 | unanswerable | yes | unavailable / not_in_documents | 0.722 | - | - | - | 4.07 |
| U2 | unanswerable | yes | unavailable / not_in_documents | 0.704 | - | - | - | 4.09 |
| U3 | unanswerable | yes | unavailable / low_relevance | 0.483 | - | - | - | 0.04 |
| U4 | unanswerable | yes | unavailable / not_in_documents | 0.630 | - | - | - | 2.17 |
| U5 | unanswerable | yes | unavailable / not_in_documents | 0.753 | - | - | - | 3.94 |

## Answers

### A1 - How many days of paid annual leave do full-time employees get?

Full-time employees receive 24 days of paid annual leave per calendar year [S1].

Cited: employee_handbook.md  

### A2 - What is the API rate limit on the Pro plan?

The API rate limit on the Pro plan is 600 requests per minute [S1].

Cited: product_faq.txt  

### A3 - What is the minimum length for company account passwords?

The minimum length for company account passwords is 14 characters [S1].

Cited: information_security_policy.pdf  

### A4 - What is the daily meal allowance when travelling internationally?

The daily meal allowance when travelling internationally is USD 60 per day [S1].

Cited: travel_expense_policy.md  

### A5 - How soon after a trip do I need to submit my expense claim?

You need to submit your expense claim within 30 days of the end of the trip [S1].

Cited: travel_expense_policy.md  

### A6 - How long is data retained for Enterprise customers?

Enterprise customers have data retention of 7 years, and this retention is configurable [S1].

Cited: product_faq.txt  

### A7 - When someone leaves the company, how quickly must their access be revoked?

When someone leaves the company, all of their access must be revoked within 24 hours of their last working day [S1].

Cited: information_security_policy.pdf  

### A8 - Since the Starter plan includes SAML single sign-on, how do we turn it on?

I could not find this information in the provided documents.


### A9 - What is the maximum nightly hotel rate I can claim on a business trip to Paris?

The maximum nightly hotel rate for international travel is USD 250 per night, which applies to cities such as London, New York, Tokyo, and Singapore. Since Paris is an international destination, the same limit applies. [S1]

Cited: travel_expense_policy.md  

### M1 - If I lose my company laptop on a business trip, how quickly must I report it, and will I have to pay for it?

You must report the loss of your company laptop within 4 hours of discovering it, as required by the Information Security Policy [S5]. If it is reported within the required time, the first loss is not charged to you, provided it was reported within the required time [S1]. For any further loss within 24 months, you may be charged up to 50% of the device's depreciated value [S1]. Therefore, you will not have to pay for the first loss if reported promptly, but could be charged up to 50% of the depreciated value for subsequent losses within 24 months [S1].

Cited: information_security_policy.pdf, employee_handbook.md  

### M2 - How many days a week can I work remotely, and what do I need to do before using public Wi-Fi in a cafe?

You can work remotely for up to 3 days per week [S2]. Before using public Wi-Fi in a cafe, you must be connected to the company VPN, NimbusConnect [S1].

Cited: employee_handbook.md, information_security_policy.pdf  

### M3 - Who needs to approve an international trip with an estimated cost of USD 4,000?

An international trip with an estimated cost of USD 4,000 requires approval from both the manager and the department head, as international trips require approval from both the manager and the department head [S1]. Additionally, because the total estimated cost exceeds USD 3,000, approval from the Chief Financial Officer (CFO) is also required [S1].

Cited: travel_expense_policy.md  

### M4 - Compare the data retention period and uptime SLA of the Pro and Enterprise plans.

The data retention period for Pro is 1 year, while Enterprise retains data for 7 years (configurable) [S2]. The uptime SLA for Pro is 99.9% monthly, and Enterprise includes a 99.95% monthly uptime SLA [S1].

Cited: product_faq.txt, product_faq.txt  

### M5 - For a 9-hour flight to Tokyo, which cabin class can I book and what is the nightly hotel limit there?

For a 9-hour flight to Tokyo, premium economy may be booked because the flight duration is 6 hours or more [S2]. The nightly hotel limit in Tokyo is USD 250 per night [S1].

Cited: travel_expense_policy.md, travel_expense_policy.md  

### U1 - How many weeks of parental leave do employees get?

I could not find this information in the provided documents.


### U2 - Who is the CEO of Nimbus Analytics?

I could not find this information in the provided documents.


### U3 - Who won the 2022 FIFA World Cup?

I could not find this information in the provided documents.


### U4 - What percentage of salary does the company contribute to the employee retirement fund?

I could not find this information in the provided documents.


### U5 - How many employees does Nimbus Analytics have?

I could not find this information in the provided documents.

