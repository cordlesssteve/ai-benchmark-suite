# Operations Manual

## Daily Operations

### Health Monitoring

#### Automated Health Checks

The system provides comprehensive health monitoring:

```bash
# Overall system health
curl http://localhost:8000/health

# Detailed component status
curl http://localhost:8000/status

# Prometheus metrics
curl http://localhost:9090/metrics

# Service-specific health
docker-compose -f docker-compose.production.yml ps
```

#### Dashboard Monitoring

Access the monitoring dashboard:

```bash
# Streamlit dashboard
python scripts/monitoring_dashboard.py

# Or access via browser: http://localhost:8501
```

Key metrics to monitor daily:

- **System Resource Usage** (CPU, Memory, Disk)
- **Evaluation Success Rate** (>95% target)
- **Cache Hit Rate** (>70% optimal)
- **API Response Times** (<2s average)
- **Active Evaluations** (within capacity limits)

### Log Monitoring

#### Critical Log Patterns

Monitor for these patterns in logs:

```bash
# Error patterns
docker-compose logs | grep -E "(ERROR|CRITICAL|FATAL)"

# Performance warnings
docker-compose logs | grep -E "(SLOW|TIMEOUT|MEMORY)"

# Authentication issues
docker-compose logs | grep -E "(AUTH|UNAUTHORIZED|FORBIDDEN)"
```

#### Log Rotation

Logs are automatically rotated, but monitor sizes:

```bash
# Check log sizes
docker system df

# Clean old logs if needed
docker system prune -f
```

## Performance Management

### Resource Optimization

#### Memory Management

Monitor and optimize memory usage:

```python
# Check memory optimizer status
from model_interfaces.memory_optimizer import MemoryOptimizer

optimizer = MemoryOptimizer()
stats = optimizer.get_memory_stats()
print(f"Memory usage: {stats['process_memory_mb']} MB")
print(f"Optimization active: {stats['optimization_active']}")

# Force garbage collection if needed
optimizer.force_cleanup()
```

#### Cache Optimization

Monitor cache performance:

```python
# Check cache statistics
from model_interfaces.result_cache_manager import ResultCacheManager

cache = ResultCacheManager()
stats = cache.get_cache_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
print(f"Cache size: {stats['cache_size_mb']} MB")

# Clear cache if hit rate is low
if stats['hit_rate'] < 0.3:
    cache.clear_cache()
```

### Scaling Operations

#### Auto-scaling Triggers

Scale up when:
- CPU usage > 80% for 5+ minutes
- Memory usage > 85% for 5+ minutes
- Queue length > 10 pending evaluations
- Response time > 5 seconds average

Scale down when:
- CPU usage < 30% for 15+ minutes
- Memory usage < 50% for 15+ minutes
- Queue length = 0 for 10+ minutes

#### Manual Scaling

```bash
# Scale API services
docker-compose -f docker-compose.production.yml up -d --scale api=3

# Scale evaluation workers
# Edit MAX_PARALLEL_EVALUATIONS in .env.production
# Restart services
docker-compose -f docker-compose.production.yml restart api
```

## Security Operations

### Access Management

#### User Management

```bash
# Add new user (via API)
curl -X POST http://localhost:8000/auth/users \
  -H "Authorization: Bearer ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "new_user",
    "email": "user@company.com",
    "role": "user"
  }'

# Disable user
curl -X PUT http://localhost:8000/auth/users/user_id/disable \
  -H "Authorization: Bearer ADMIN_TOKEN"
```

#### API Key Rotation

```bash
# Generate new API key
curl -X POST http://localhost:8000/auth/api-keys \
  -H "Authorization: Bearer ADMIN_TOKEN" \
  -d '{"name": "production-key", "expires_in_days": 90}'

# Revoke old API key
curl -X DELETE http://localhost:8000/auth/api-keys/key_id \
  -H "Authorization: Bearer ADMIN_TOKEN"
```

### Security Monitoring

#### Failed Authentication Attempts

```bash
# Monitor failed logins
docker-compose logs api | grep "authentication failed"

# Check for brute force attempts
docker-compose logs api | grep "rate limit exceeded"
```

#### Suspicious Activity

Monitor for:
- Unusual evaluation patterns
- High-frequency API calls
- Resource exhaustion attempts
- Unauthorized access attempts

### Backup and Recovery

#### Automated Backups

Daily backup verification:

```bash
# Check backup status
ls -la /backups/

# Verify latest backup integrity
scripts/verify_backup.sh $(ls -t /backups/ | head -1)
```

#### Manual Backup

```bash
# Create immediate backup
scripts/create_backup.sh emergency-$(date +%Y%m%d-%H%M)

# Backup specific components
docker exec postgres pg_dump ai_benchmark_prod > manual-db-backup.sql
docker exec redis redis-cli --rdb /data/manual-redis-backup.rdb
```

#### Recovery Procedures

**Database Recovery:**

```bash
# Stop services
docker-compose -f docker-compose.production.yml down

# Restore database
docker run --rm -v $(pwd)/backups:/backups postgres:13 \
  psql -h postgres -U benchmark_user -d ai_benchmark_prod < /backups/latest-db.sql

# Restart services
docker-compose -f docker-compose.production.yml up -d
```

**Configuration Recovery:**

```bash
# Restore configuration
tar -xzf config-backup.tar.gz

# Update secrets
python scripts/restore_secrets.py config-backup/secrets.encrypted
```

## Incident Response

### Severity Levels

#### Critical (P1) - Immediate Response Required
- Complete service outage
- Data corruption
- Security breach
- Evaluation failure rate > 50%

#### High (P2) - Response within 1 hour
- Partial service degradation
- Performance degradation > 50%
- Authentication issues
- Evaluation failure rate 20-50%

#### Medium (P3) - Response within 4 hours
- Minor performance issues
- Non-critical feature failures
- Evaluation failure rate 10-20%

#### Low (P4) - Response within 24 hours
- Cosmetic issues
- Documentation updates
- Minor optimization opportunities

### Incident Response Procedures

#### Immediate Response (First 15 minutes)

1. **Assess Severity**
   ```bash
   # Quick health check
   curl http://localhost:8000/health

   # Check system resources
   docker stats --no-stream

   # Review recent logs
   docker-compose logs --tail=100
   ```

2. **Stabilize System**
   ```bash
   # If memory issues
   docker exec api python -c "
   from model_interfaces.memory_optimizer import MemoryOptimizer
   MemoryOptimizer().force_cleanup()
   "

   # If evaluation backlog
   curl -X POST http://localhost:8000/api/v1/queue/clear

   # If cache issues
   docker exec redis redis-cli FLUSHALL
   ```

3. **Communication**
   - Update status page
   - Notify stakeholders
   - Document incident start time

#### Investigation Phase (15-60 minutes)

1. **Collect Diagnostics**
   ```bash
   # Generate diagnostic report
   scripts/generate_diagnostics.sh incident-$(date +%Y%m%d-%H%M)

   # Check performance metrics
   curl http://localhost:8000/api/v1/debug/performance

   # Analyze evaluation patterns
   curl http://localhost:8000/api/v1/analytics/recent-patterns
   ```

2. **Root Cause Analysis**
   - Review system changes
   - Analyze performance trends
   - Check external dependencies
   - Examine evaluation patterns

#### Resolution Phase

1. **Implement Fix**
   ```bash
   # Configuration changes
   scripts/update_config.sh incident-response

   # Service restart if needed
   docker-compose -f docker-compose.production.yml restart service-name

   # Database fixes if needed
   scripts/repair_database.sh
   ```

2. **Verify Resolution**
   ```bash
   # Health verification
   scripts/verify_system_health.sh

   # Performance testing
   scripts/run_smoke_tests.sh

   # End-to-end validation
   curl -X POST http://localhost:8000/api/v1/evaluate \
     -d '{"model_name":"test-model","task":"test","language":"python"}'
   ```

### Post-Incident Procedures

#### Documentation

1. **Incident Report**
   - Timeline of events
   - Root cause analysis
   - Resolution steps
   - Lessons learned

2. **Update Runbooks**
   - Add new troubleshooting steps
   - Update monitoring alerts
   - Improve automation scripts

#### Prevention

1. **Monitoring Improvements**
   ```bash
   # Add new alerts based on incident
   scripts/update_alerts.sh incident-$(date +%Y%m%d)

   # Enhance monitoring
   scripts/add_monitoring.sh new-metric-name
   ```

2. **System Hardening**
   - Update resource limits
   - Improve error handling
   - Add circuit breakers
   - Enhance logging

## Maintenance Windows

### Scheduled Maintenance

#### Weekly Maintenance (Sunday 2-4 AM)

```bash
# Maintenance script
scripts/weekly_maintenance.sh
```

Tasks performed:
- Database optimization and VACUUM
- Cache cleanup and optimization
- Log rotation and archival
- Resource usage analysis
- Security scan
- Performance benchmarking

#### Monthly Maintenance (First Sunday 2-6 AM)

```bash
# Extended maintenance script
scripts/monthly_maintenance.sh
```

Additional tasks:
- System updates and patches
- Configuration review
- Capacity planning analysis
- Security assessment
- Backup verification
- Performance optimization

### Emergency Maintenance

#### Preparation

```bash
# Create maintenance backup
scripts/pre_maintenance_backup.sh

# Verify rollback procedures
scripts/verify_rollback.sh
```

#### Execution

```bash
# Enter maintenance mode
curl -X POST http://localhost:8000/admin/maintenance-mode

# Perform maintenance tasks
scripts/emergency_maintenance.sh

# Exit maintenance mode
curl -X DELETE http://localhost:8000/admin/maintenance-mode
```

#### Verification

```bash
# Post-maintenance verification
scripts/post_maintenance_verification.sh

# Performance validation
scripts/validate_performance.sh
```

## Monitoring and Alerting Configuration

### Alert Thresholds

#### System Alerts

```yaml
# High CPU usage
cpu_usage > 85% for 5 minutes

# High memory usage
memory_usage > 90% for 5 minutes

# Disk space low
disk_usage > 85%

# High network latency
network_latency > 1000ms for 2 minutes
```

#### Application Alerts

```yaml
# Evaluation failure rate
evaluation_failure_rate > 20% for 5 minutes

# Cache hit rate low
cache_hit_rate < 30% for 10 minutes

# API response time high
api_response_time > 5000ms for 5 minutes

# Queue depth high
evaluation_queue_depth > 20 for 10 minutes
```

### Custom Dashboards

#### Executive Dashboard

Key metrics for management:
- System uptime percentage
- Evaluation success rate trends
- Cost per evaluation
- User activity metrics

#### Technical Dashboard

Detailed metrics for operations:
- Resource utilization trends
- Performance bottleneck analysis
- Error rate by component
- Cache performance analytics

## Performance Optimization Guidelines

### Model Performance

#### HuggingFace Optimization

```python
# Optimize model loading
config = {
    "device_map": "auto",
    "torch_dtype": "float16",
    "low_cpu_mem_usage": True,
    "load_in_8bit": True
}
```

#### API Performance

```python
# Optimize API responses
app.add_middleware(
    GZipMiddleware,
    minimum_size=1000
)

# Connection pooling
SQLALCHEMY_POOL_SIZE = 20
SQLALCHEMY_MAX_OVERFLOW = 30
```

### Database Performance

#### Query Optimization

```sql
-- Add indexes for common queries
CREATE INDEX idx_evaluations_created_at ON evaluations(created_at);
CREATE INDEX idx_evaluations_status ON evaluations(status);
CREATE INDEX idx_cache_keys_hash ON cache_keys(key_hash);
```

#### Connection Management

```python
# Optimize database connections
DATABASE_CONFIG = {
    "pool_size": 20,
    "max_overflow": 30,
    "pool_timeout": 30,
    "pool_recycle": 3600
}
```

## Troubleshooting Guide

### Common Issues and Solutions

#### High Memory Usage

**Symptoms:**
- Slow evaluation performance
- System warnings about memory
- OOM errors in logs

**Investigation:**
```bash
# Check memory usage
docker stats --no-stream
free -h

# Check process memory
ps aux --sort=-%mem | head -10
```

**Solutions:**
1. Enable memory optimization
2. Reduce parallel evaluations
3. Clear model cache
4. Restart services

#### Cache Performance Issues

**Symptoms:**
- Low cache hit rates
- Slow evaluation times
- High database load

**Investigation:**
```bash
# Check cache statistics
curl http://localhost:8000/api/v1/cache/stats

# Check Redis memory
docker exec redis redis-cli INFO memory
```

**Solutions:**
1. Optimize cache key generation
2. Increase cache size
3. Clear corrupted cache entries
4. Tune cache TTL values

#### API Timeout Issues

**Symptoms:**
- Request timeouts
- High response times
- Client connection errors

**Investigation:**
```bash
# Check API performance
curl -w "@curl-format.txt" http://localhost:8000/health

# Check connection pool
curl http://localhost:8000/api/v1/debug/connections
```

**Solutions:**
1. Increase timeout values
2. Scale API services
3. Optimize database queries
4. Enable connection pooling

This operations manual provides comprehensive guidance for daily operations, incident response, and maintenance of the AI Benchmark Suite production environment.