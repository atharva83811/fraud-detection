## Testing

- Verified model predictions using known fraud and non-fraud samples
- Validated API input handling with FastAPI schema enforcement
- Tested edge cases (zero and high-value transactions)
- Performed end-to-end testing via React dashboard

### Threshold Tuning

Due to class imbalance, I tuned the classification threshold to improve recall.
Lowering the threshold from 0.5 to 0.25 increased fraud recall from 49% to 82%
while maintaining high precision.