Read SPEC.md first and follow it.
Work on ONE phase at a time. Do not start the next phase until I confirm.
Never fabricate results. All metrics, charts, and statistics must be computed from the actual dataset.
No hard-coded dataset statistics anywhere.
Use random_state=42 for reproducibility.
Fit scalers and encoders on the training set only (no leakage).
Use Streamlit caching (st.cache_data / st.cache_resource) for heavy work.
Show clear, user-friendly error messages in the UI.
If something fails: identify the exact error, explain why, fix only the failing part, and give me the exact command to rerun. Do not rewrite unrelated files.
After each phase report: what was completed, files changed, how to run it, what I should understand, what remains.
Explain each important module in beginner-friendly language: what, why, input, internals, output, and what to say in viva.
