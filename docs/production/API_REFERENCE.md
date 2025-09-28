# API Reference

## Base URL

Production: `https://your-domain.com/api/v1`
Development: `http://localhost:8000/api/v1`

## Authentication

All API endpoints require authentication using JWT tokens or API keys.

### Get Access Token

```http
POST /auth/token
Content-Type: application/json

{
  "username": "your_username",
  "password": "your_password"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### Using Authentication

Include the token in the Authorization header:

```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

## Core Evaluation Endpoints

### Single Model Evaluation

Evaluate a single model on a specific task.

```http
POST /evaluate
Authorization: Bearer {token}
Content-Type: application/json

{
  "model_name": "gpt-3.5-turbo",
  "task": "humaneval",
  "language": "python",
  "parameters": {
    "temperature": 0.7,
    "max_tokens": 512,
    "top_p": 1.0
  },
  "use_cache": true,
  "timeout_seconds": 300
}
```

**Parameters:**
- `model_name` (string, required): Model identifier
- `task` (string, required): Evaluation task name
- `language` (string, required): Programming language
- `parameters` (object, optional): Model-specific parameters
- `use_cache` (boolean, optional): Enable result caching (default: true)
- `timeout_seconds` (integer, optional): Evaluation timeout (default: 300)

**Response:**
```json
{
  "evaluation_id": "eval_123456789",
  "status": "running",
  "created_at": "2024-01-15T10:30:00Z",
  "estimated_completion": "2024-01-15T10:35:00Z",
  "websocket_url": "ws://localhost:8000/ws/evaluations/eval_123456789"
}
```

### Suite Evaluation

Evaluate multiple models across multiple tasks and languages.

```http
POST /evaluate/suite
Authorization: Bearer {token}
Content-Type: application/json

{
  "models": ["gpt-3.5-turbo", "gpt-4", "claude-3-sonnet"],
  "tasks": ["humaneval", "mbpp", "codecontests"],
  "languages": ["python", "javascript", "java"],
  "suite_config": {
    "parallel_executions": 4,
    "use_cache": true,
    "timeout_per_evaluation": 600,
    "continue_on_failure": true
  },
  "parameters": {
    "gpt-3.5-turbo": {
      "temperature": 0.7,
      "max_tokens": 512
    },
    "gpt-4": {
      "temperature": 0.5,
      "max_tokens": 1024
    }
  }
}
```

**Parameters:**
- `models` (array, required): List of model identifiers
- `tasks` (array, required): List of evaluation tasks
- `languages` (array, required): List of programming languages
- `suite_config` (object, optional): Suite-wide configuration
- `parameters` (object, optional): Model-specific parameters

**Response:**
```json
{
  "suite_id": "suite_987654321",
  "status": "running",
  "total_evaluations": 27,
  "completed_evaluations": 0,
  "failed_evaluations": 0,
  "created_at": "2024-01-15T10:30:00Z",
  "estimated_completion": "2024-01-15T12:30:00Z",
  "websocket_url": "ws://localhost:8000/ws/suites/suite_987654321",
  "individual_evaluations": [
    {
      "evaluation_id": "eval_123456789",
      "model": "gpt-3.5-turbo",
      "task": "humaneval",
      "language": "python",
      "status": "pending"
    }
  ]
}
```

### Get Evaluation Status

```http
GET /evaluate/{evaluation_id}
Authorization: Bearer {token}
```

**Response:**
```json
{
  "evaluation_id": "eval_123456789",
  "status": "completed",
  "model_name": "gpt-3.5-turbo",
  "task": "humaneval",
  "language": "python",
  "created_at": "2024-01-15T10:30:00Z",
  "started_at": "2024-01-15T10:30:05Z",
  "completed_at": "2024-01-15T10:34:23Z",
  "execution_time_seconds": 258,
  "result": {
    "pass_at_k": {
      "pass@1": 0.73,
      "pass@5": 0.85,
      "pass@10": 0.91
    },
    "total_problems": 164,
    "solved_problems": 120,
    "success_rate": 0.73,
    "average_attempts": 2.3,
    "detailed_results": [
      {
        "problem_id": "HumanEval/0",
        "passed": true,
        "attempts": 1,
        "execution_time": 1.2
      }
    ]
  },
  "cache_hit": false,
  "resource_usage": {
    "peak_memory_mb": 1024,
    "cpu_time_seconds": 45.2,
    "gpu_utilization": 0.85
  }
}
```

### Get Suite Status

```http
GET /evaluate/suite/{suite_id}
Authorization: Bearer {token}
```

**Response:**
```json
{
  "suite_id": "suite_987654321",
  "status": "completed",
  "total_evaluations": 27,
  "completed_evaluations": 25,
  "failed_evaluations": 2,
  "created_at": "2024-01-15T10:30:00Z",
  "completed_at": "2024-01-15T12:15:33Z",
  "total_execution_time_seconds": 6333,
  "summary": {
    "overall_pass_at_1": 0.68,
    "overall_pass_at_5": 0.82,
    "overall_pass_at_10": 0.89,
    "model_rankings": [
      {
        "model": "gpt-4",
        "average_pass_at_1": 0.81,
        "rank": 1
      },
      {
        "model": "claude-3-sonnet",
        "average_pass_at_1": 0.73,
        "rank": 2
      }
    ],
    "task_performance": [
      {
        "task": "humaneval",
        "average_pass_at_1": 0.75,
        "best_model": "gpt-4"
      }
    ]
  },
  "evaluations": [
    {
      "evaluation_id": "eval_123456789",
      "model": "gpt-3.5-turbo",
      "task": "humaneval",
      "language": "python",
      "status": "completed",
      "pass_at_1": 0.73
    }
  ]
}
```

## Real-time Updates

### WebSocket Connection

Connect to WebSocket for real-time evaluation updates:

```javascript
// Single evaluation updates
const ws = new WebSocket('ws://localhost:8000/ws/evaluations/eval_123456789');

// Suite evaluation updates
const ws = new WebSocket('ws://localhost:8000/ws/suites/suite_987654321');

ws.onmessage = function(event) {
  const update = JSON.parse(event.data);
  console.log('Status update:', update);
};
```

**Message Format:**
```json
{
  "type": "status_update",
  "evaluation_id": "eval_123456789",
  "status": "running",
  "progress": 0.45,
  "current_problem": 74,
  "total_problems": 164,
  "intermediate_results": {
    "solved_so_far": 33,
    "current_pass_rate": 0.72
  },
  "timestamp": "2024-01-15T10:32:15Z"
}
```

## System Management

### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "4.0.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "model_interfaces": "healthy"
  },
  "metrics": {
    "active_evaluations": 3,
    "queue_depth": 5,
    "cache_hit_rate": 0.78
  }
}
```

### System Status

```http
GET /status
Authorization: Bearer {token}
```

**Response:**
```json
{
  "system": {
    "uptime_seconds": 86400,
    "version": "4.0.0",
    "environment": "production"
  },
  "performance": {
    "cpu_usage_percent": 45.2,
    "memory_usage_percent": 67.3,
    "disk_usage_percent": 23.1
  },
  "evaluations": {
    "total_completed": 15420,
    "total_failed": 234,
    "success_rate": 0.985,
    "average_execution_time": 185.5
  },
  "cache": {
    "hit_rate": 0.782,
    "size_mb": 2048,
    "entries": 12543
  },
  "queue": {
    "pending_evaluations": 5,
    "running_evaluations": 3,
    "max_parallel": 8
  }
}
```

### Metrics

```http
GET /metrics
Authorization: Bearer {token}
```

Returns Prometheus-formatted metrics for monitoring.

## Configuration Management

### Get Configuration

```http
GET /config
Authorization: Bearer {token}
```

**Response:**
```json
{
  "models": {
    "available_models": [
      "gpt-3.5-turbo",
      "gpt-4",
      "claude-3-sonnet",
      "claude-3-opus"
    ],
    "default_parameters": {
      "temperature": 0.7,
      "max_tokens": 512
    }
  },
  "tasks": {
    "available_tasks": [
      "humaneval",
      "mbpp",
      "codecontests"
    ],
    "supported_languages": {
      "humaneval": ["python", "javascript", "java", "cpp"],
      "mbpp": ["python"],
      "codecontests": ["python", "java", "cpp"]
    }
  },
  "limits": {
    "max_parallel_evaluations": 8,
    "max_timeout_seconds": 1800,
    "max_suite_size": 100
  }
}
```

### Update Configuration

```http
PUT /config
Authorization: Bearer {admin_token}
Content-Type: application/json

{
  "limits": {
    "max_parallel_evaluations": 12,
    "max_timeout_seconds": 2400
  }
}
```

## Cache Management

### Cache Statistics

```http
GET /cache/stats
Authorization: Bearer {token}
```

**Response:**
```json
{
  "hit_rate": 0.782,
  "cache_size_mb": 2048.5,
  "total_entries": 12543,
  "total_requests": 65432,
  "cache_hits": 51168,
  "cache_misses": 14264,
  "language_breakdown": {
    "python": {
      "cached_results": 8543,
      "hit_rate": 0.79
    },
    "javascript": {
      "cached_results": 2341,
      "hit_rate": 0.73
    }
  },
  "model_breakdown": {
    "gpt-3.5-turbo": {
      "cached_results": 5432,
      "hit_rate": 0.81
    },
    "gpt-4": {
      "cached_results": 3421,
      "hit_rate": 0.76
    }
  }
}
```

### Clear Cache

```http
DELETE /cache
Authorization: Bearer {admin_token}

# Optional: Clear specific model/task/language
DELETE /cache?model=gpt-3.5-turbo&task=humaneval&language=python
```

## Queue Management

### Queue Status

```http
GET /queue/status
Authorization: Bearer {token}
```

**Response:**
```json
{
  "pending_evaluations": 5,
  "running_evaluations": 3,
  "max_parallel": 8,
  "queue_depth": 5,
  "estimated_wait_time_seconds": 420,
  "pending_items": [
    {
      "evaluation_id": "eval_234567890",
      "model": "gpt-4",
      "task": "humaneval",
      "language": "python",
      "priority": 1,
      "estimated_start": "2024-01-15T10:37:00Z"
    }
  ],
  "running_items": [
    {
      "evaluation_id": "eval_123456789",
      "model": "gpt-3.5-turbo",
      "task": "mbpp",
      "language": "python",
      "started_at": "2024-01-15T10:30:00Z",
      "progress": 0.65
    }
  ]
}
```

### Priority Evaluation

```http
POST /evaluate/priority
Authorization: Bearer {admin_token}
Content-Type: application/json

{
  "model_name": "gpt-4",
  "task": "humaneval",
  "language": "python",
  "priority": 1
}
```

### Cancel Evaluation

```http
DELETE /evaluate/{evaluation_id}
Authorization: Bearer {token}
```

## Analytics and Reporting

### Performance Analytics

```http
GET /analytics/performance
Authorization: Bearer {token}

# Optional query parameters:
# ?start_date=2024-01-01&end_date=2024-01-31
# ?model=gpt-3.5-turbo
# ?task=humaneval
# ?language=python
```

**Response:**
```json
{
  "time_period": {
    "start": "2024-01-01T00:00:00Z",
    "end": "2024-01-31T23:59:59Z"
  },
  "total_evaluations": 1543,
  "success_rate": 0.987,
  "average_execution_time": 185.2,
  "performance_trends": [
    {
      "date": "2024-01-01",
      "evaluations": 52,
      "success_rate": 0.98,
      "avg_time": 178.5
    }
  ],
  "model_performance": [
    {
      "model": "gpt-4",
      "evaluations": 523,
      "success_rate": 0.994,
      "avg_pass_at_1": 0.81,
      "avg_execution_time": 220.5
    }
  ],
  "task_performance": [
    {
      "task": "humaneval",
      "evaluations": 654,
      "avg_pass_at_1": 0.73,
      "avg_execution_time": 195.2
    }
  ]
}
```

### Cost Analysis

```http
GET /analytics/costs
Authorization: Bearer {token}
```

**Response:**
```json
{
  "total_cost_usd": 1250.75,
  "cost_breakdown": {
    "gpt-3.5-turbo": {
      "evaluations": 523,
      "total_tokens": 2543210,
      "cost_usd": 381.48
    },
    "gpt-4": {
      "evaluations": 234,
      "total_tokens": 1234567,
      "cost_usd": 740.74
    }
  },
  "cost_per_evaluation": {
    "gpt-3.5-turbo": 0.73,
    "gpt-4": 3.16
  },
  "monthly_trend": [
    {
      "month": "2024-01",
      "cost_usd": 1250.75,
      "evaluations": 757
    }
  ]
}
```

## Error Handling

### Error Response Format

All API errors follow this format:

```json
{
  "error": {
    "code": "EVALUATION_FAILED",
    "message": "Evaluation failed due to model timeout",
    "details": {
      "evaluation_id": "eval_123456789",
      "model": "gpt-3.5-turbo",
      "timeout_seconds": 300
    },
    "timestamp": "2024-01-15T10:35:00Z",
    "request_id": "req_987654321"
  }
}
```

### HTTP Status Codes

- `200` - Success
- `201` - Created
- `202` - Accepted (for async operations)
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `409` - Conflict
- `422` - Unprocessable Entity
- `429` - Rate Limited
- `500` - Internal Server Error
- `503` - Service Unavailable

### Common Error Codes

- `INVALID_MODEL` - Specified model is not supported
- `INVALID_TASK` - Specified task is not available
- `INVALID_LANGUAGE` - Language not supported for the task
- `EVALUATION_TIMEOUT` - Evaluation exceeded timeout limit
- `RESOURCE_EXHAUSTED` - System resources are at capacity
- `CACHE_ERROR` - Cache operation failed
- `AUTHENTICATION_FAILED` - Invalid credentials
- `RATE_LIMIT_EXCEEDED` - Too many requests
- `CONFIGURATION_ERROR` - Invalid configuration parameters

## Rate Limiting

Rate limits are applied per API key:

- **Standard users**: 100 requests per hour
- **Premium users**: 1000 requests per hour
- **Enterprise users**: 10000 requests per hour

Rate limit headers are included in responses:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642248600
```

## Pagination

List endpoints support pagination:

```http
GET /evaluations?page=2&limit=50&sort=created_at&order=desc
```

**Response:**
```json
{
  "items": [...],
  "pagination": {
    "page": 2,
    "limit": 50,
    "total_items": 1543,
    "total_pages": 31,
    "has_next": true,
    "has_prev": true
  }
}
```

## SDKs and Client Libraries

### Python SDK

```python
from ai_benchmark_sdk import BenchmarkClient

client = BenchmarkClient(
    api_key="your_api_key",
    base_url="https://your-domain.com/api/v1"
)

# Single evaluation
result = client.evaluate(
    model="gpt-3.5-turbo",
    task="humaneval",
    language="python"
)

# Suite evaluation
suite = client.evaluate_suite(
    models=["gpt-3.5-turbo", "gpt-4"],
    tasks=["humaneval", "mbpp"],
    languages=["python"]
)
```

### JavaScript SDK

```javascript
import { BenchmarkClient } from '@ai-benchmark/sdk';

const client = new BenchmarkClient({
  apiKey: 'your_api_key',
  baseUrl: 'https://your-domain.com/api/v1'
});

// Single evaluation
const result = await client.evaluate({
  model: 'gpt-3.5-turbo',
  task: 'humaneval',
  language: 'python'
});

// Suite evaluation
const suite = await client.evaluateSuite({
  models: ['gpt-3.5-turbo', 'gpt-4'],
  tasks: ['humaneval', 'mbpp'],
  languages: ['python']
});
```