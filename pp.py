from qiskit_ibm_runtime import QiskitRuntimeService


print("BEFORE:", QiskitRuntimeService.saved_accounts())
QiskitRuntimeService.delete_account(channel="ibm_quantum_platform")
print("AFTER DELETE:", QiskitRuntimeService.saved_accounts())

QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",
    token="LxwVq5-g3yYyWaijj03aY38DsMjXi37u-RejKoXNVjM0",
    overwrite=True,
    set_as_default=True,
)

service = QiskitRuntimeService()
print("saved accounts:", QiskitRuntimeService.saved_accounts())
print("instances:", service.instances())
print("active account:", service.active_account())
print("active instance:", service.active_instance())
print("visible backends:", [b.name for b in service.backends()])

from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService(instance="bnn")
print(service.instances())
print([b.name for b in service.backends()])

for name in ["ibm_kingston", "ibm_fez", "ibm_marrakesh"]:
    try:
        print(name, "->", service.backend(name))
    except Exception as e:
        print(name, "FAILED:", repr(e))