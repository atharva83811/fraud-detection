import { useEffect, useState } from "react";
import api from "./api";

function App() {
    const [amount, setAmount] = useState("");
    const [result, setResult] = useState(null);
    const [metrics, setMetrics] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // 🔹 Fetch model metrics ONCE when app loads
    useEffect(() => {
        api.get("/metrics")
            .then((res) => setMetrics(res.data))
            .catch(() => console.error("Failed to load metrics"));
    }, []);

    const submit = async () => {
        try {
            setLoading(true);
            setError(null);

            const res = await api.post("/predict", {
                Amount: Number(amount),
            });

            setResult(res.data);
        } catch (err) {
            setError("Prediction failed. Check backend logs.", err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="p-8 max-w-xl mx-auto space-y-6">
            <h1 className="text-2xl font-bold">
                Fraud Detection Dashboard
            </h1>

            {/* 🔹 Model Metrics */}
            {metrics && (
                <div className="p-4 border rounded bg-black-50">
                    <h2 className="font-semibold mb-2">Model Metrics</h2>
                    <p><strong>Model:</strong> {metrics.model}</p>
                    <p><strong>Precision:</strong> {metrics.precision}</p>
                    <p><strong>Recall:</strong> {metrics.recall}</p>
                    <p><strong>F1 Score:</strong> {metrics.f1}</p>
                    {metrics.threshold && (
                        <p><strong>Threshold:</strong> {metrics.threshold}</p>
                    )}
                </div>
            )}

            {/* 🔹 Prediction Form */}
            <div className="space-y-3">
                <input
                    type="number"
                    placeholder="Transaction Amount"
                    value={amount}
                    onChange={(e) => setAmount(e.target.value)}
                    className="border p-2 w-full rounded"
                />

                <button
                    onClick={submit}
                    disabled={loading}
                    className="bg-blue-600 text-white px-4 py-2 rounded disabled:opacity-50"
                >
                    {loading ? "Predicting..." : "Predict"}
                </button>
            </div>

            {/* 🔹 Errors */}
            {error && (
                <p className="text-red-600">{error}</p>
            )}

            {/* 🔹 Prediction Result */}
            {result && (
                <div className="p-4 border rounded">
                    <p>
                        <strong>Fraud Probability:</strong>{" "}
                        {result.fraud_probability}
                    </p>
                    <p>
                        <strong>Verdict:</strong>{" "}
                        {result.verdict}
                    </p>
                </div>
            )}
        </div>
    );
}

export default App;
