# Ringtail Production Deployment Guide

## Overview

Ringtail is a lattice-based threshold signature scheme implementation designed for integration with the Lux Network consensus protocol. This guide covers production deployment considerations and improvements made for production readiness.

## Version: v0.1.0

### Key Features
- Two-round threshold signature protocol
- Post-quantum secure (based on LWE)
- Optimized for consensus integration
- Configurable parameters
- Production-ready error handling

## Production Improvements

### 1. Module Structure
- Updated module name to `github.com/luxfi/ringtail`
- Proper package organization for production use

### 2. Configuration Management
- External configuration support via JSON files
- Environment variable overrides
- Default configurations for different network types

### 3. Error Handling
- Comprehensive error wrapping with context
- Panic recovery mechanisms
- Structured error types for better debugging

### 4. Network Resilience
- Connection retry logic
- Timeout configurations
- Concurrent-safe socket management

### 5. Consensus Integration
- Configurable consensus timeouts
- Node ID management
- Signature concurrency controls

## Deployment

### Prerequisites
- Go 1.19 or higher
- Network connectivity between signature parties
- Sufficient system resources (see below)

### Resource Requirements

#### Minimum (5-party network)
- CPU: 2 cores
- RAM: 4GB
- Network: 100 Mbps
- Storage: 1GB

#### Recommended (21-party mainnet)
- CPU: 8 cores
- RAM: 16GB
- Network: 1 Gbps
- Storage: 10GB

### Configuration

Create a `config.json` file:

```json
{
  "network": {
    "listen_port": 8080,
    "peer_addresses": [
      "node1.lux.network:8080",
      "node2.lux.network:8080"
    ],
    "timeout_seconds": 30,
    "max_retries": 3
  },
  "signature": {
    "party_count": 21,
    "threshold": 15,
    "m": 8,
    "n": 7,
    "key_size": 32
  },
  "consensus": {
    "enabled": true,
    "consensus_timeout_ms": 9630,
    "signature_timeout_ms": 2000,
    "node_id": "NodeID-xxx"
  }
}
```

### Running

#### Single Node
```bash
./ringtail <party_id> <iterations> <party_count> --config config.json
```

#### Docker Deployment
```dockerfile
FROM golang:1.19-alpine AS builder
WORKDIR /app
COPY . .
RUN go build -o ringtail .

FROM alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /root/
COPY --from=builder /app/ringtail .
COPY config.json .
CMD ["./ringtail"]
```

### Monitoring

Key metrics to monitor:
- Signature generation time (target: < 2s)
- Network latency between parties
- Memory usage
- CPU utilization
- Failed signature attempts

### Security Considerations

1. **Key Management**
   - Store keys in secure hardware (HSM)
   - Rotate keys periodically
   - Use separate keys for test/mainnet

2. **Network Security**
   - Use TLS for all communications
   - Implement IP whitelisting
   - Monitor for anomalous behavior

3. **Access Control**
   - Limit access to signature nodes
   - Use strong authentication
   - Audit all access attempts

## Integration with Lux Consensus

### Interface
```go
type ThresholdSigner interface {
    // Initialize the signer with configuration
    Init(config *Config) error
    
    // Generate a threshold signature
    Sign(message []byte, signers []int) (*Signature, error)
    
    // Verify a threshold signature
    Verify(signature *Signature, message []byte) bool
    
    // Get signer status
    Status() (*SignerStatus, error)
}
```

### Consensus Parameters
- Mainnet: 21 nodes, 15 threshold, 9.63s timeout
- Testnet: 11 nodes, 7 threshold, 6.3s timeout
- Local: 5 nodes, 3 threshold, 3.69s timeout

## Troubleshooting

### Common Issues

1. **Connection Failures**
   - Check firewall rules
   - Verify peer addresses
   - Ensure network connectivity

2. **Signature Timeouts**
   - Increase timeout values
   - Check network latency
   - Verify all parties are online

3. **Performance Issues**
   - Enable performance profiling
   - Check system resources
   - Optimize network topology

### Debug Mode
```bash
RINGTAIL_DEBUG=true ./ringtail <args>
```

## Future Improvements

1. **Performance Optimizations**
   - Implement signature aggregation
   - Optimize polynomial operations
   - Add GPU acceleration support

2. **Operational Features**
   - Prometheus metrics export
   - Health check endpoints
   - Automatic key rotation

3. **Consensus Enhancements**
   - Direct consensus integration
   - Signature caching
   - Batch signature support

## Support

For issues and questions:
- GitHub Issues: https://github.com/luxfi/ringtail/issues
- Security: security@lux.network

## License

Apache 2.0 - See LICENSE file