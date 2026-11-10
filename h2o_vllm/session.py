"""Request/layer isolated H2O policy state for benchmark integrations."""
from dataclasses import dataclass, field
from .policy import H2OPolicy, H2OPolicyConfig

@dataclass
class H2OSessions:
    config: H2OPolicyConfig
    _policies: dict[tuple[str, str], H2OPolicy] = field(default_factory=dict)

    def get(self, request_id: str, layer_name: str) -> H2OPolicy:
        if not request_id or not layer_name:
            raise ValueError("Both request_id and layer_name are mandatory")
        return self._policies.setdefault((request_id, layer_name), H2OPolicy(self.config))

    def finish_request(self, request_id: str) -> None:
        for key in tuple(self._policies):
            if key[0] == request_id:
                del self._policies[key]

    def clear(self) -> None:
        self._policies.clear()
