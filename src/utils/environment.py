"""
Environment detection and configuration manager for debug vs production modes
"""
import os
import json
import platform
import socket

class EnvironmentManager:
    """Manages debug vs production environment settings"""
    
    def __init__(self):
        self.is_windows = platform.system() == "Windows"
        self.is_hpc = self._detect_hpc_environment()
        self.debug_mode = self._should_use_debug_mode()
        
    def _detect_hpc_environment(self):
        """Detect if running on HPC environment"""
        # Common HPC indicators
        hpc_indicators = [
            'SLURM_JOB_ID',  # SLURM scheduler
            'PBS_JOBID',     # PBS scheduler
            'LSB_JOBID',     # LSF scheduler
            'SGE_JOB_ID',    # SGE scheduler
        ]
        
        # Check for HPC environment variables
        for indicator in hpc_indicators:
            if os.getenv(indicator):
                return True
                
        # Check hostname patterns (common HPC naming)
        hostname = socket.gethostname().lower()
        hpc_patterns = ['compute', 'node', 'gpu', 'hpc', 'cluster']
        
        for pattern in hpc_patterns:
            if pattern in hostname:
                return True
                
        return False
    
    def _should_use_debug_mode(self):
        """Determine if should use debug mode"""
        # Force debug mode if explicitly set
        if os.getenv('TFEW_DEBUG', '').lower() in ['true', '1', 'yes']:
            return True
            
        # Force production mode if explicitly set
        if os.getenv('TFEW_PRODUCTION', '').lower() in ['true', '1', 'yes']:
            return False
            
        # Auto-detect: Windows = debug, HPC = production
        if self.is_windows:
            return True
        elif self.is_hpc:
            return False
        else:
            # Default to debug for other environments
            return True
    
    def get_config_overrides(self):
        """Get configuration overrides based on environment"""
        config_file = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'debug_config.json')
        
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Debug config file not found at {config_file}")
            return {}
        
        if self.debug_mode:
            settings = config_data.get('debug_settings', {})
            print("[DEBUG MODE] Running with reduced resources for quick testing")
        else:
            settings = config_data.get('production_settings', {})
            print("[PRODUCTION MODE] Running with full resources")
            
        return settings
    
    def should_skip_deepspeed(self):
        """Check if DeepSpeed should be skipped"""
        return self.debug_mode or self.is_windows
    
    def get_environment_info(self):
        """Get environment information for logging"""
        return {
            'platform': platform.system(),
            'hostname': socket.gethostname(),
            'is_hpc': self.is_hpc,
            'is_windows': self.is_windows,
            'debug_mode': self.debug_mode,
            'python_version': platform.python_version(),
        }

# Global instance
env_manager = EnvironmentManager()

def get_env_manager():
    """Get the global environment manager instance"""
    return env_manager

def print_environment_info():
    """Print environment information"""
    env_info = env_manager.get_environment_info()
    print("=" * 50)
    print("ENVIRONMENT INFORMATION")
    print("=" * 50)
    for key, value in env_info.items():
        print(f"{key:15}: {value}")
    print("=" * 50)
