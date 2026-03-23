from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",
    token="hi_nP_YYoZwuwxhjDpuPWzHxUQmoVdtbWGsCQ7aG9s2_",
    instance="crn:v1:bluemix:public:quantum-computing:us-east:a/a9f8252b6ce94df9a42c20f24a5f730b:a199a303-3447-4eba-95fa-b9d4078a6096::.",
    overwrite=True,
    set_as_default=True,
)



from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService()
service.delete_account(channel="ibm_quantum_platform")
service.save_account(
    channel="ibm_quantum_platform",
    token="hi_nP_YYoZwuwxhjDpuPWzHxUQmoVdtbWGsCQ7aG9s2_",
    instance="crn:v1:bluemix:public:quantum-computing:us-east:a/a9f8252b6ce94df9a42c20f24a5f730b:a199a303-3447-4eba-95fa-b9d4078a6096::",
    overwrite=True,
    set_as_default=True,
)
print(service.backends())
print(QiskitRuntimeService.saved_accounts())
