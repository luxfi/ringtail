package utils

import (
	"fmt"
	"runtime"
)

// Error wraps an error with additional context
type Error struct {
	Op   string // operation
	Kind string // category of error
	Err  error  // underlying error
}

func (e *Error) Error() string {
	if e.Err == nil {
		return fmt.Sprintf("%s: %s", e.Op, e.Kind)
	}
	return fmt.Sprintf("%s: %s: %v", e.Op, e.Kind, e.Err)
}

// Unwrap returns the underlying error
func (e *Error) Unwrap() error {
	return e.Err
}

// WrapError creates a new Error with context
func WrapError(op string, kind string, err error) error {
	return &Error{Op: op, Kind: kind, Err: err}
}

// PanicHandler recovers from panics and converts them to errors
func PanicHandler(op string) {
	if r := recover(); r != nil {
		_, file, line, _ := runtime.Caller(2)
		err := fmt.Errorf("panic in %s at %s:%d: %v", op, file, line, r)
		panic(WrapError(op, "panic", err))
	}
}