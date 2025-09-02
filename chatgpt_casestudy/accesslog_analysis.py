import requests
import json
from datetime import datetime
from collections import Counter
import re
import os
import glob
from prettytable import PrettyTable
import ipaddress

from pdb import set_trace

# Bot configurations
BOT_CONFIGS = {
    "ChatGPT-User": {
        "url": "https://openai.com/chatgpt-user.json",
        "user_agent_pattern": "ChatGPT-User"
    },
    "OAI-SearchBot": {
        "url": "https://openai.com/searchbot.json",
        "user_agent_pattern": "OAI-SearchBot"
    },
    "GPTBot": {
        "url": "https://openai.com/gptbot.json",
        "user_agent_pattern": "GPTBot"
    },
    "PerplexityBot": {
        "url": "https://www.perplexity.com/perplexitybot.json",
        "user_agent_pattern": "PerplexityBot"
    },
    "Perplexity-User": {
        "url": "https://www.perplexity.com/perplexity-user.json",
        "user_agent_pattern": "Perplexity-User"
    }
}

def fetch_bot_ips(bot_name):
    """Fetch bot IPs from OpenAI's JSON file and save locally if new version"""
    if bot_name not in BOT_CONFIGS:
        print(f"Unknown bot: {bot_name}")
        return None
    
    url = BOT_CONFIGS[bot_name]["url"]
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Create directory if it doesn't exist
        ip_dir = "bot_ips"
        os.makedirs(ip_dir, exist_ok=True)
        
        # Create filename with creation time
        # Handle both formats: with and without Z suffix
        creation_time_str = data['creationTime']
        if creation_time_str.endswith('Z'):
            creation_time = datetime.strptime(creation_time_str[:-1], "%Y-%m-%dT%H:%M:%S.%f")
        else:
            creation_time = datetime.strptime(creation_time_str, "%Y-%m-%dT%H:%M:%S.%f")
            
        filename = f"{bot_name}_{creation_time.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(ip_dir, filename)
        
        # Check if we already have this version
        existing_files = glob.glob(os.path.join(ip_dir, f"{bot_name}_*.json"))
        for existing_file in existing_files:
            try:
                with open(existing_file, 'r') as f:
                    existing_data = json.load(f)
                    if existing_data['creationTime'] == data['creationTime']:
                        print(f"{bot_name} IPs version {data['creationTime']} already exists")
                        return data
            except Exception as e:
                print(f"Error reading existing file {existing_file}: {e}")
        
        # Save new version
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Saved new {bot_name} IPs version to {filepath}")
        
        return data
    except Exception as e:
        print(f"Error fetching {bot_name} IPs: {e}")
        return None

def get_bot_ips_for_time(bot_name, access_time):
    """Get the appropriate bot IPs version for the given access time"""
    ip_dir = "bot_ips"
    
    # Get all IP files for this bot and sort by creation time
    ip_files = glob.glob(os.path.join(ip_dir, f"{bot_name}_*.json"))
    ip_files.sort(reverse=True)  # Sort newest first
    
    for ip_file in ip_files:
        try:
            with open(ip_file, 'r') as f:
                data = json.load(f)
                # Handle both formats: with and without Z suffix
                creation_time_str = data['creationTime']
                if creation_time_str.endswith('Z'):
                    creation_time = datetime.strptime(creation_time_str[:-1], "%Y-%m-%dT%H:%M:%S.%f")
                else:
                    creation_time = datetime.strptime(creation_time_str, "%Y-%m-%dT%H:%M:%S.%f")
                
                # Make the datetime timezone-aware if the access_time is
                if access_time.tzinfo is not None and creation_time.tzinfo is None:
                    # Use the same timezone as access_time
                    creation_time = creation_time.replace(tzinfo=access_time.tzinfo)
                    
                if creation_time <= access_time:
                    return data
        except Exception as e:
            print(f"Error reading IP file {ip_file}: {e}")
    
    return None

def parse_access_log(log_file):
    """Parse access log file and extract IPs, timestamps, and User Agents for all bots"""
    pattern = r'(?P<ip>[\d.]+) - - \[(?P<timestamp>.*?)\] "(?P<request>.*?)" (?P<status>\d+) (?P<bytes>\d+) "(?P<referer>.*?)" "(?P<user_agent>.*?)"'
    
    log_entries = []
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                match = re.match(pattern, line)
                if match:
                    data = match.groupdict()
                    user_agent = data['user_agent']
                    
                    # Determine the bot type
                    bot_name = None
                    for bot, config in BOT_CONFIGS.items():
                        if config["user_agent_pattern"].lower() in user_agent.lower():
                            bot_name = bot
                            break
                    
                    if bot_name:
                        # Convert timestamp to datetime object
                        timestamp = datetime.strptime(data['timestamp'], '%d/%b/%Y:%H:%M:%S %z')
                        
                        # Extract URL from request
                        request_parts = data['request'].split(' ')
                        url = request_parts[1] if len(request_parts) > 1 else "-"
                        
                        log_entries.append({
                            'ip': data['ip'],
                            'timestamp': timestamp,
                            'user_agent': user_agent,
                            'url': url,
                            'bot_name': bot_name
                        })
    except Exception as e:
        print(f"Error reading log file {log_file}: {e}")
    
    return log_entries

def validate_bot_ips(ip, bot_data):
    """Validate if an IP matches bot IPs using CIDR ranges"""
    if not bot_data:
        return False, None, None
    
    try:
        ip_addr = ipaddress.ip_address(ip)
        for prefix in bot_data['prefixes']:
            network = ipaddress.ip_network(prefix['ipv4Prefix'])
            if ip_addr in network:
                # Get first and last IP in the range
                first_ip = network[0]
                last_ip = network[-1]
                ip_range = f"{prefix['ipv4Prefix']} ({first_ip} - {last_ip})"
                return True, prefix['ipv4Prefix'], ip_range
    except Exception as e:
        print(f"Error validating IP {ip}: {e}")
        return False, None, None
    
    return False, None, None

def analyze_logs(log_files):
    """Analyze log files and return all entries"""
    all_entries = []
    
    for log_file in log_files:
        print(f"Processing {log_file}...")
        entries = parse_access_log(log_file)
        all_entries.extend(entries)
    
    return all_entries

def get_log_files_after_date(start_date):
    """Get all log files after the specified date"""
    log_dir = "./llmbot_access_logs"
    log_files = glob.glob(os.path.join(log_dir, "access_log_*.log"))
    
    # Filter files after start_date
    filtered_files = []
    for log_file in log_files:
        try:
            # Extract date from filename (format: access_log_YYYYMMDD.log)
            date_str = log_file.split('_')[-1].split('.')[0]
            file_date = datetime.strptime(date_str, '%Y%m%d')
            if file_date >= start_date:
                filtered_files.append(log_file)
        except Exception as e:
            print(f"Error processing file {log_file}: {e}")
    
    return sorted(filtered_files)

def main():
    # Get start date (March 27, 2025)
    start_date = datetime(2025, 3, 27)
    
    # Fetch latest IPs for all bots and save if new
    for bot_name in BOT_CONFIGS:
        fetch_bot_ips(bot_name)
    
    # Get all log files after start date
    log_files = get_log_files_after_date(start_date)
    
    if not log_files:
        print("No log files found after March 27, 2025")
        # Still show summary with zeros
        print("\n\nSummary of all bot accesses:")
        summary_table = PrettyTable()
        summary_table.field_names = ["Bot Type", "Total Requests", "Valid IPs", "Invalid IPs"]
        
        for bot_name in BOT_CONFIGS:
            summary_table.add_row([bot_name, 0, 0, 0])
        
        print(summary_table)
        return
    
    print(f"Found {len(log_files)} log files to analyze")
    
    # Get all bot access log entries
    bot_entries = analyze_logs(log_files)
    
    if not bot_entries:
        print("\nNo bot access logs found!")
        # Still show summary with zeros
        print("\n\nSummary of all bot accesses:")
        summary_table = PrettyTable()
        summary_table.field_names = ["Bot Type", "Total Requests", "Valid IPs", "Invalid IPs"]
        
        for bot_name in BOT_CONFIGS:
            summary_table.add_row([bot_name, 0, 0, 0])
        
        print(summary_table)
        return
    
    # Sort entries by timestamp to process them chronologically
    bot_entries.sort(key=lambda x: x['timestamp'])
    
    # Group entries by bot type
    bot_groups = {}
    for entry in bot_entries:
        if entry['bot_name'] not in bot_groups:
            bot_groups[entry['bot_name']] = []
        bot_groups[entry['bot_name']].append(entry)

    # Remove debug breakpoint
    # set_trace()
    
    # Process each bot type
    for bot_name, entries in bot_groups.items():
        print(f"\n\n{'='*50}")
        print(f"Processing {bot_name} ({len(entries)} entries)")
        print(f"{'='*50}")
        
        # Create and populate the table for this bot's entries
        table = PrettyTable()
        table.field_names = ["IP Address", "Access Time", "URL", "User Agent", "Valid IP?", "Bot IP Version", "Matched IP Range"]
        table.align["IP Address"] = "l"
        table.align["Access Time"] = "l"
        table.align["URL"] = "l"
        table.align["User Agent"] = "l"
        table.align["Valid IP?"] = "c"
        table.align["Bot IP Version"] = "c"
        table.align["Matched IP Range"] = "l"
        
        invalid_ips = []
        
        # Get all IP versions for this bot
        ip_dir = "bot_ips"
        ip_files = glob.glob(os.path.join(ip_dir, f"{bot_name}_*.json"))
        ip_versions = []
        
        for ip_file in ip_files:
            try:
                with open(ip_file, 'r') as f:
                    data = json.load(f)
                    # Handle both formats: with and without Z suffix
                    creation_time_str = data['creationTime']
                    if creation_time_str.endswith('Z'):
                        creation_time = datetime.strptime(creation_time_str[:-1], "%Y-%m-%dT%H:%M:%S.%f")
                    else:
                        creation_time = datetime.strptime(creation_time_str, "%Y-%m-%dT%H:%M:%S.%f")
                    
                    ip_versions.append({
                        'time': creation_time,
                        'data': data
                    })
            except Exception as e:
                print(f"Error reading IP file {ip_file}: {e}")
        
        # Sort IP versions by time (newest first)
        ip_versions.sort(key=lambda x: x['time'], reverse=True)
        
        # Process entries
        current_ip_version_index = 0
        for entry in entries:
            # Find appropriate IP version for this entry's timestamp
            while (current_ip_version_index < len(ip_versions) - 1 and 
                   entry['timestamp'].replace(tzinfo=None) < ip_versions[current_ip_version_index]['time']):
                current_ip_version_index += 1
            
            bot_data = ip_versions[current_ip_version_index]['data'] if current_ip_version_index < len(ip_versions) else None
            is_valid, cidr, ip_range = validate_bot_ips(entry['ip'], bot_data)
            
            if not is_valid:
                invalid_ips.append(entry)
            
            table.add_row([
                entry['ip'],
                entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                entry['url'],
                "..." + entry['user_agent'][-50:] if len(entry['user_agent']) > 40 else entry['user_agent'],
                "✓" if is_valid else "✗",
                ip_versions[current_ip_version_index]['time'].strftime('%Y-%m-%d %H:%M:%S') if current_ip_version_index < len(ip_versions) else "N/A",
                ip_range if ip_range else "N/A"
            ])
        
        print(f"\n{bot_name} Access Logs:")
        print(table)
        
        if invalid_ips:
            print(f"\n{bot_name} IPs not matching ranges:")
            invalid_table = PrettyTable()
            invalid_table.field_names = ["IP Address", "Access Time", "URL", "User Agent"]
            invalid_table.align["IP Address"] = "l"
            invalid_table.align["Access Time"] = "l"
            invalid_table.align["URL"] = "l"
            invalid_table.align["User Agent"] = "l"
            
            for entry in invalid_ips:
                invalid_table.add_row([
                    entry['ip'],
                    entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    entry['url'],
                    entry['user_agent']
                ])
            
            print(invalid_table)
            print(f"\nTotal invalid {bot_name} IPs found: {len(invalid_ips)}")
        else:
            print(f"\nAll {bot_name} IPs match expected ranges!")
    
    # Overall summary
    print("\n\nSummary of all bot accesses:")
    summary_table = PrettyTable()
    summary_table.field_names = ["Bot Type", "Total Requests", "Valid IPs", "Invalid IPs"]
    
    # Make sure all configured bots are represented in the summary
    for bot_name in BOT_CONFIGS:
        if bot_name in bot_groups:
            entries = bot_groups[bot_name]
            total = len(entries)
            
            # Count invalid IPs
            invalid = 0
            for entry in entries:
                try:
                    # Get appropriate IP data for the timestamp, converting timestamps consistently
                    bot_data = get_bot_ips_for_time(bot_name, entry['timestamp'])
                    is_valid = validate_bot_ips(entry['ip'], bot_data)[0]
                    if not is_valid:
                        invalid += 1
                except Exception as e:
                    # Count as invalid if there's an error
                    invalid += 1
                    print(f"Error validating {bot_name} IP {entry['ip']}: {e}")
            
            valid = total - invalid
        else:
            # No entries for this bot
            total, valid, invalid = 0, 0, 0
        
        summary_table.add_row([
            bot_name,
            total,
            valid,
            invalid
        ])
    
    print(summary_table)

if __name__ == "__main__":
    main()
