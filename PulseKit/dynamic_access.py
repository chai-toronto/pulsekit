class DynamicAttributeAccess:
    """Base class to enable dot notation access to dictionary-like data"""
    
    def __init__(self, data_dict):
        self._data = data_dict
    
    def __getattr__(self, name):
        """Enable dot notation access to dictionary keys"""
        if name in self._data:
            value = self._data[name]
            # If the value is also a dict, wrap it for chaining
            if hasattr(value, '__getitem__') and hasattr(value, 'keys'):
                return DynamicAttributeAccess(value)
            return value
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __iter__(self):
        """Allows 'for x in wrapper:'"""
        return iter(self._data)

    def __len__(self):
        """Allows 'len(wrapper)'"""
        return len(self._data)

    def values(self):
        """Allows 'wrapper.values()' to return the objects"""
        return self._data.values()
    
    def __getitem__(self, key):
        """Still support dictionary-style access"""
        return self._data[key]
    
    def __contains__(self, key):
        return key in self._data
    
    def keys(self):
        return self._data.keys()
    
    def values(self):
        return self._data.values()
    
    def items(self):
        return self._data.items()


class SessionsWrapper(DynamicAttributeAccess):
    """Wrapper specifically for sessions to handle session IDs as attributes"""
    
    def __getattr__(self, name):
        # 1. Try finding the name exactly as typed
        if name in self._data:
            session_id = name
        # 2. If not found, and it starts with '_', try stripping it (e.g. _1 -> 1)
        elif name.startswith('_') and name[1:] in self._data:
            session_id = name[1:]
        else:
            # If neither works, raise the error
            raise AttributeError(f"Session '{name}' (mapped to '{name}') not found. Available keys: {list(self._data.keys())}")

        # If we reach here, we found a valid session_id
        session_obj = self._data[session_id]
        
        # Wrap the signals dict if it exists (Auto-upgrade to SignalsWrapper)
        if hasattr(session_obj, 'signals') and not isinstance(session_obj.signals, SignalsWrapper):
            # Handle case where .signals might already be a wrapper or a raw dict
            raw_signals = session_obj.signals._data if hasattr(session_obj.signals, '_data') else session_obj.signals
            session_obj.signals = SignalsWrapper(raw_signals)
            
        return session_obj

class SignalsWrapper(DynamicAttributeAccess):
    """Wrapper specifically for signals to handle signal names as attributes"""
    
    def __getattr__(self, name):
        # Handle signal names that might have special characters
        if name in self._data:
            return self._data[name]
        
        # Try replacing underscores with the original format if needed
        for key in self._data.keys():
            if key.replace('_', '').replace('-', '') == name.replace('_', '').replace('-', ''):
                return self._data[key]
        
        raise AttributeError(f"Signal '{name}' not found")


def add_dynamic_access(cls):
    """Decorator to add dynamic access to a class"""
    
    original_init = cls.__init__
    
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Wrap sessions dict with dynamic access
        if hasattr(self, 'sessions'):
            self.sessions = SessionsWrapper(self.sessions)
    
    cls.__init__ = new_init
    return cls