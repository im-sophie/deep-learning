import abc

class MemoryCleanupScheduleBase(abc.ABC):
    @abc.abstractmethod
    def on_should_cleanup(self, memory_buffer, runner_context):
        pass

    @abc.abstractmethod
    def on_cleanup(self, memory_buffer, runner_context):
        pass

    def should_cleanup(self, memory_buffer, runner_context):
        return self.on_should_cleanup(memory_buffer, runner_context)

    def cleanup(self, memory_buffer, runner_context):
        return self.on_cleanup(memory_buffer, runner_context)