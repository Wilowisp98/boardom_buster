import time
import asyncio
import httpx
from typing import List, Tuple, Dict
from logger import get_logger

logger = get_logger(__name__)

async def fetch_bgg_api(game_ids: List[int], sleep_time: int = 3, max_retries: int = 3, retry_delay: int = 5) -> Dict[str, int]:
    """
    Fetch game data from BoardGameGeek API and return rate limit statistics
    Returns: Dictionary with 'rate_limited' (bool) and 'rate_limit_retries' (int)
    """
    endpoint = "https://boardgamegeek.com/xmlapi2/thing"
    params = {
        "id": ",".join(map(str, game_ids)),
        "type": "boardgame",
        "stats": 1
    }
    
    result = {
        "rate_limited": False,
        "rate_limit_retries": 0  # Count how many times we had to sleep due to rate limits
    }
    
    async with httpx.AsyncClient() as client:
        for attempt in range(max_retries):
            try:
                response = await client.get(endpoint, params=params)
                
                if response.status_code == 200:
                    # Successful request
                    await asyncio.sleep(sleep_time)
                    return result
                
                elif response.status_code == 429:
                    # Rate limited
                    logger.warning(f"Rate limited. Retry {attempt+1}/{max_retries}")
                    result["rate_limited"] = True
                    result["rate_limit_retries"] += 1
                    await asyncio.sleep(retry_delay * (attempt + 1))
                else:
                    # Other error
                    logger.error(f"Error fetching data: {response.status_code}")
                    await asyncio.sleep(retry_delay)
            
            except Exception as e:
                logger.error(f"Exception during fetch: {e}")
                await asyncio.sleep(retry_delay)
        
        logger.error(f"Failed to fetch data after {max_retries} attempts")
        return result

async def benchmark_config(batch_size: int = 30, max_chunk_size: int = 10, sleep_time: int = 3, retry_delay: int = 5) -> Tuple[int, int, float]:
    """
    Benchmark a specific configuration and return rate limit stats and execution time
    Returns: Tuple of (total_rate_limits, total_rate_limit_retries, execution_time)
    """
    current_id = 1
    max_id = 300  # Adjust as needed for benchmark duration
    total_rate_limits = 0
    total_rate_limit_retries = 0
    
    logger.info(f"Benchmarking: batch_size={batch_size}, max_chunk_size={max_chunk_size}, sleep_time={sleep_time}, retry_delay={retry_delay}")
    start_time = time.time()
    
    while current_id <= max_id:
        batch_ids = list(range(current_id, current_id + batch_size))
        chunks_ids = [batch_ids[i:i+max_chunk_size] for i in range(0, len(batch_ids), max_chunk_size)]
        
        tasks = []
        for chunk_ids in chunks_ids:
            tasks.append(fetch_bgg_api(chunk_ids, sleep_time, 3, retry_delay))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count rate limit stats
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task error: {result}")
                total_rate_limits += 1
            elif result["rate_limited"]:
                total_rate_limits += 1
                total_rate_limit_retries += result["rate_limit_retries"]
                
        current_id += batch_size
    
    execution_time = time.time() - start_time
    return total_rate_limits, total_rate_limit_retries, execution_time

async def optimize_configuration(initial_config=(20, 10, 3, 5), iterations=10, max_id=300):
    """
    Use a gradient descent-like approach to find optimal configuration.
    
    Args:
        initial_config: Starting point (batch_size, max_chunk_size, sleep_time, retry_delay)
        iterations: Maximum number of optimization iterations
        max_id: Maximum ID to use for benchmarks during optimization
        
    Returns:
        Best configuration found
    """
    # Save original max_id from benchmark_config
    original_max_id = globals().get('benchmark_config').__defaults__[0]  # Get the default max_id
    
    # Use a smaller max_id for faster optimization iterations
    # We'll temporarily monkey patch the function
    async def benchmark_with_custom_max_id(*args, **kwargs):
        current_id = 1
        custom_max_id = max_id  # Use the smaller max_id for optimization
        total_rate_limits = 0
        total_rate_limit_retries = 0
        
        batch_size, max_chunk_size, sleep_time, retry_delay = args
        
        logger.info(f"Quick benchmark: batch_size={batch_size}, max_chunk_size={max_chunk_size}, sleep_time={sleep_time}, retry_delay={retry_delay}")
        start_time = time.time()
        
        while current_id <= custom_max_id:
            batch_ids = list(range(current_id, current_id + batch_size))
            chunks_ids = [batch_ids[i:i+max_chunk_size] for i in range(0, len(batch_ids), max_chunk_size)]
            
            tasks = []
            for chunk_ids in chunks_ids:
                tasks.append(fetch_bgg_api(chunk_ids, sleep_time, 3, retry_delay))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count rate limit stats
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Task error: {result}")
                    total_rate_limits += 1
                elif result["rate_limited"]:
                    total_rate_limits += 1
                    total_rate_limit_retries += result["rate_limit_retries"]
                    
            current_id += batch_size
        
        execution_time = time.time() - start_time
        return total_rate_limits, total_rate_limit_retries, execution_time
    
    current_config = initial_config
    current_batch, current_chunk, current_sleep, current_delay = current_config
    
    # Bounds for each parameter to ensure valid values
    param_bounds = [
        (5, 100),   # batch_size: min 5, max 50
        (10, 10),   # max_chunk_size: min 1, max 30
        (1, 10),   # sleep_time: min 1, max 10
        (1, 15)    # retry_delay: min 1, max 15
    ]
    
    # Step sizes for each parameter
    param_steps = [5, 2, 1, 1]
    
    # Weight factors: How to balance execution time vs rate limiting
    time_weight = 1.0
    retry_weight = 0.2  # Lower gives more priority to execution speed
    
    logger.info(f"Starting optimization with initial config: {current_config}")
    
    # Initial benchmark to establish baseline
    rate_limits, rate_retries, exec_time = await benchmark_with_custom_max_id(*current_config)
    current_score = exec_time + (rate_retries * retry_weight)
    
    logger.info(f"Initial score: {current_score:.2f} (time: {exec_time:.2f}s, retries: {rate_retries})")
    
    best_configs = [(current_config, rate_limits, rate_retries, exec_time, current_score)]
    
    for iteration in range(iterations):
        logger.info(f"Optimization iteration {iteration+1}/{iterations}")
        best_score = current_score
        best_config = current_config
        improved = False
        
        # For each parameter, try adjusting up and down
        for param_idx, (param_name, param_value, param_step) in enumerate([
            ("batch_size", current_batch, param_steps[0]),
            ("max_chunk_size", current_chunk, param_steps[1]),
            ("sleep_time", current_sleep, param_steps[2]),
            ("retry_delay", current_delay, param_steps[3])
        ]):
            # Try increasing the parameter
            param_min, param_max = param_bounds[param_idx]
            if param_value + param_step <= param_max:
                test_config = list(current_config)
                test_config[param_idx] = param_value + param_step
                test_config = tuple(test_config)
                
                rate_limits, rate_retries, exec_time = await benchmark_with_custom_max_id(*test_config)
                test_score = exec_time + (rate_retries * retry_weight)
                
                logger.info(f"  {param_name}+{param_step}={test_config[param_idx]}: score={test_score:.2f} (time: {exec_time:.2f}s, retries: {rate_retries})")
                best_configs.append((test_config, rate_limits, rate_retries, exec_time, test_score))
                
                if test_score < best_score:
                    best_score = test_score
                    best_config = test_config
                    improved = True
            
            # Try decreasing the parameter
            if param_value - param_step >= param_min:
                test_config = list(current_config)
                test_config[param_idx] = param_value - param_step
                test_config = tuple(test_config)
                
                rate_limits, rate_retries, exec_time = await benchmark_with_custom_max_id(*test_config)
                test_score = exec_time + (rate_retries * retry_weight)
                
                logger.info(f"  {param_name}-{param_step}={test_config[param_idx]}: score={test_score:.2f} (time: {exec_time:.2f}s, retries: {rate_retries})")
                best_configs.append((test_config, rate_limits, rate_retries, exec_time, test_score))
                
                if test_score < best_score:
                    best_score = test_score
                    best_config = test_config
                    improved = True
        
        # Update current configuration if improved
        if improved:
            logger.info(f"Found better configuration: {best_config} (score: {best_score:.2f})")
            current_config = best_config
            current_batch, current_chunk, current_sleep, current_delay = current_config
            current_score = best_score
        else:
            logger.info(f"No improvement found. Stopping optimization.")
            break
    
    # Sort by score (best first)
    best_configs.sort(key=lambda x: x[4])
    
    # Print summary of all tested configurations
    logger.info("=" * 85)
    logger.info("Optimization Results (sorted by score):")
    logger.info("=" * 85)
    logger.info("| Batch | Chunk | Sleep | Retry | Rate Limits | Limit Retries | Time (s) | Score |")
    logger.info("|-------|-------|-------|-------|-------------|---------------|----------|-------|")
    
    for config, rate_limits, rate_retries, exec_time, score in best_configs:
        logger.info(f"| {config[0]:5d} | {config[1]:5d} | {config[2]:5d} | {config[3]:5d} | {rate_limits:11d} | {rate_retries:13d} | {exec_time:8.2f} | {score:5.2f} |")
    
    # Final evaluation of best configuration using original benchmark function
    logger.info("=" * 85)
    logger.info(f"Running final benchmark with best configuration: {best_configs[0][0]}")
    rate_limits, rate_retries, exec_time = await benchmark_config(*best_configs[0][0])
    
    logger.info("=" * 85)
    logger.info(f"Best configuration found: batch_size={best_configs[0][0][0]}, chunk_size={best_configs[0][0][1]}, sleep_time={best_configs[0][0][2]}, retry_delay={best_configs[0][0][3]}")
    logger.info(f"Rate limits: {rate_limits}, Rate limit retries: {rate_retries}, Execution time: {exec_time:.2f}s")
    
    return best_configs[0][0]

async def main_bgg(mode="manual"):
    if mode == "manual":
        # Manual testing of predefined configurations
        test_configs = [
            (10, 10, 3, 5),
            (20, 10, 3, 5),
            (30, 10, 3, 5),
            (30, 15, 3, 5),
            (30, 10, 2, 5),
            (30, 10, 4, 5)
        ]
        
        results = []
        for config in test_configs:
            logger.info(f"Testing configuration: {config}")
            rate_limits, rate_limit_retries, execution_time = await benchmark_config(*config)
            
            results.append((config, rate_limits, rate_limit_retries, execution_time))
            logger.info(f"Configuration {config} - Rate limits: {rate_limits}, Retries: {rate_limit_retries}, Time: {execution_time:.2f}s")
        
        # Sort results by execution time (fastest first)
        results.sort(key=lambda x: x[3])
        
        # Print summary of results
        logger.info("=" * 80)
        logger.info("Benchmark Results (sorted by execution time):")
        logger.info("=" * 80)
        logger.info("| Batch | Chunk | Sleep | Retry | Rate Limits | Limit Retries | Time (s) |")
        logger.info("|-------|-------|-------|-------|-------------|---------------|----------|")
        
        for config, rate_limits, rate_retries, exec_time in results:
            logger.info(f"| {config[0]:5d} | {config[1]:5d} | {config[2]:5d} | {config[3]:5d} | {rate_limits:11d} | {rate_retries:13d} | {exec_time:8.2f} |")
        
        # Print the best configuration
        best_config, best_rate_limits, best_rate_retries, best_time = results[0]
        logger.info("=" * 80)
        logger.info(f"Best configuration: batch_size={best_config[0]}, chunk_size={best_config[1]}, sleep_time={best_config[2]}, retry_delay={best_config[3]}")
        logger.info(f"Rate limits: {best_rate_limits}, Rate limit retries: {best_rate_retries}, Execution time: {best_time:.2f}s")
    
    elif mode == "optimize":
        # Automatic optimization
        await optimize_configuration(initial_config=(20, 10, 3, 5), iterations=8, max_id=100)

if __name__ == "__main__":
    # Choose mode: "manual" or "optimize"
    asyncio.run(main_bgg("optimize"))