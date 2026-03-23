from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()
print(service.backends())

'''
service = QiskitRuntimeService(channel="ibm_quantum_platform",
    token="LxwVq5-g3yYyWaijj03aY38DsMjXi37u-RejKoXNVjM0",
    instance="crn:v1:bluemix:public:quantum-computing:us-east:a/a9f8252b6ce94df9a42c20f24a5f730b:a199a303-3447-4eba-95fa-b9d4078a6096::",)
service.delete_account()
service.save_account(
    channel="ibm_quantum_platform",
    token="LxwVq5-g3yYyWaijj03aY38DsMjXi37u-RejKoXNVjM0",
    instance="crn:v1:bluemix:public:quantum-computing:us-east:a/a9f8252b6ce94df9a42c20f24a5f730b:a199a303-3447-4eba-95fa-b9d4078a6096::",
    overwrite=True,
    set_as_default=True,
)
print(service.backends())
print(service.saved_accounts())
'''