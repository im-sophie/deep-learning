# Internal
from ..RunnerFactoryBase import RunnerFactoryBase

class RunnerClassifierTreeToNLPresentationFactory(RunnerFactoryBase):
    def on_create_default_hyperparameter_set(self) -> HyperparameterSet:
        hyperparameter_set = HyperparameterSet()
        hyperparameter_set.add("batch_size_training", 10 ** 3)
        hyperparameter_set.add("batch_size_testing", 10 ** 3)
        hyperparameter_set.add("epoch_count", 25)
        hyperparameter_set.add("learning_rate", 0.001)
        return hyperparameter_set

    def _get_function_argument_type_permutations(self,
        max_count: int) -> List[List[TypeBase]]:
        assert max_count > 0

        if max_count == 1:
            return [TypeBool(), TypeAbstract(), TypeInt()]
        else:
            next_results = self._get_function_argument_type_permutations(max_count - 1)
            current_results = list(next_results)

            for i in self._get_function_argument_type_permutations(1):
                for j in next_results:
                    current_results.append([i] + j)

            return current_results

    def _get_function_type_permutations(self,
        max_argument_count: int,
        max_type_count: int) -> List[TypeFunction]:
        assert max_argument_count > 0
        assert max_type_count >= 0

        results = []

        for i in self._get_function_argument_type_permutations(1):
            for j in self._get_function_argument_type_permutations(max_argument_count):
                results.append(TypeFunction(i, j))

        if len(results) > max_type_count:
            return random.sample(results, max_type_count)
        else:
            return results

    def _create_scope(self) -> Scope:
        scope = Scope()

        for i in range(self.hyperparameter_set["boolean_symbol_count"]):
            scope["b{0}".format(i)] = TypeBool()

        for i in range(self.hyperparameter_set["abstract_symbol_count"]):
            scope["a{0}".format(i)] = TypeAbstract()

        for i in range(self.hyperparameter_set["int_symbol_count"]):
            scope["i{0}".format(i)] = TypeInt()

        for i, j in enumerate(self._get_function_type_permutations(self.hyperparameter_set["max_function_argument_count"], hyperparameter_set["function_symbol_count"])):
            scope["f{0}".format(i)] = j

        return scope

    def _create_synthesizer(self,
        scope: Scope) -> TreeSynthesizerBase:
        return TreeSynthesizerStochastic(
            scope,
            self.hyperparameter_set["max_value_depth"],
            self.hyperparameter_set["tree_synthesis_probability"]
        )

    def _create_network(self,
        hyperparameter_set: HyperparameterSet) -> OptimizedModule:
        return OptimizedSequential(
            nn.Linear(
                in_features = 256,
                out_features = 256
            ),
            optimizer_factory = OptimizedSequential.optimizer_factory_adam,
            learning_rate = hyperparameter_set["learning_rate"]
        )

    def on_create_runner(self,
        hyperparameter_set: HyperparameterSet,
        tensorboard_output_dir: Optional[str]) -> RunnerBase:
        return RunnerClassifierTreeToNLPresentation(
            hyperparameter_set,
            self._create_network(hyperparameter_set),
            self._create_synthesizer(
                self._create_scope()
            )
        )
