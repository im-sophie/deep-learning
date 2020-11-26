from .MemoryCleanupScheduleBase import MemoryCleanupScheduleBase

class MemoryCleanupScheduleMonteCarlo(MemoryCleanupScheduleBase):
    def on_should_cleanup(self, memory_buffer, runner_context):
        return True
    
    def on_cleanup(self, memory_buffer, runner_context):
        del memory_buffer[:]
