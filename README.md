# Log2Row
A command-line tool that detects, extracts log templates, and structures logs to in-process database, leveraging template patterns generated by GPT-4.

## What it does?
If you have several kinds of logs are mixed together (samples from [loghub](https://github.com/logpai/loghub/blob/master/Apache/Apache_2k.log)):

```text
  ↳ head -n 25 loghub/Apache/Apache_2k.log
[Sun Dec 04 04:47:44 2005] [notice] workerEnv.init() ok /etc/httpd/conf/workers2.properties
[Sun Dec 04 04:47:44 2005] [error] mod_jk child workerEnv in error state 6
[Sun Dec 04 04:51:08 2005] [notice] jk2_init() Found child 6725 in scoreboard slot 10
[Sun Dec 04 04:51:09 2005] [notice] jk2_init() Found child 6726 in scoreboard slot 8
[Sun Dec 04 04:51:09 2005] [notice] jk2_init() Found child 6728 in scoreboard slot 6
[Sun Dec 04 04:51:14 2005] [notice] workerEnv.init() ok /etc/httpd/conf/workers2.properties
[Sun Dec 04 04:51:14 2005] [notice] workerEnv.init() ok /etc/httpd/conf/workers2.properties
[Sun Dec 04 04:51:14 2005] [notice] workerEnv.init() ok /etc/httpd/conf/workers2.properties
[Sun Dec 04 04:51:18 2005] [error] mod_jk child workerEnv in error state 6
[Sun Dec 04 04:51:18 2005] [error] mod_jk child workerEnv in error state 6
[Sun Dec 04 04:51:18 2005] [error] mod_jk child workerEnv in error state 6
[Sun Dec 04 04:51:37 2005] [notice] jk2_init() Found child 6736 in scoreboard slot 10
[Sun Dec 04 04:51:38 2005] [notice] jk2_init() Found child 6733 in scoreboard slot 7
[Sun Dec 04 04:51:38 2005] [notice] jk2_init() Found child 6734 in scoreboard slot 9
[Sun Dec 04 04:51:52 2005] [notice] workerEnv.init() ok /etc/httpd/conf/workers2.properties
[Sun Dec 04 04:51:52 2005] [notice] workerEnv.init() ok /etc/httpd/conf/workers2.properties
[Sun Dec 04 04:51:55 2005] [error] mod_jk child workerEnv in error state 6
[Sun Dec 04 04:52:04 2005] [notice] jk2_init() Found child 6738 in scoreboard slot 6
[Sun Dec 04 04:52:04 2005] [notice] jk2_init() Found child 6741 in scoreboard slot 9
[Sun Dec 04 04:52:05 2005] [notice] jk2_init() Found child 6740 in scoreboard slot 7
[Sun Dec 04 04:52:05 2005] [notice] jk2_init() Found child 6737 in scoreboard slot 8
[Sun Dec 04 04:52:12 2005] [notice] workerEnv.init() ok /etc/httpd/conf/workers2.properties
[Sun Dec 04 04:52:12 2005] [notice] workerEnv.init() ok /etc/httpd/conf/workers2.properties
[Sun Dec 04 04:52:12 2005] [notice] workerEnv.init() ok /etc/httpd/conf/workers2.properties
[Sun Dec 04 04:52:15 2005] [error] mod_jk child workerEnv in error state 6
```

Log2Row could detect and extract several log templates, it would take a while, all extracted log would be stored into an embedded in-memory DB [DuckDB](http://duckdb.org/docs/archive/0.9.0/), and opens an IPython REPL:

```ipython
  ↳ OPENAI_API_KEY="***" python -m log2row run "loghub/Apache/*.log" --gpt-base https://api.openai.com/v1 --max-lines 0 --buf-size 1MB --threshold 0.9
Python 3.11.5 (main, Aug 24 2023, 15:09:45) [Clang 14.0.3 (clang-1403.0.22.14.1)]
Type 'copyright', 'credits' or 'license' for more information
IPython 8.16.0 -- An enhanced Interactive Python. Type '?' for help.

use variable `result` to get the result

In [1]: result.sql('show tables')
Out[1]:
┌───────────────────────────────────────────────────────────────────────────────┐
│                                     name                                      │
│                                    varchar                                    │
├───────────────────────────────────────────────────────────────────────────────┤
│ ^\[(.+)\] \[error\] \[client (.+)\] Directory index forbidden by rule: (.+)$  │
│ ^\[(.+)\] \[error\] jk2_init\(\) Can't find child (\d+) in scoreboard$        │
│ ^\[(.+)\] \[notice\] jk2_init\(\) Found child (\d+) in scoreboard slot (\d+)$ │
└───────────────────────────────────────────────────────────────────────────────┘

In [2]: for table in result.sql('show tables').fetchall():
   ...:     print(table[0])
   ...:     table = table[0].replace('"', '""')
   ...:     print(result.sql(f'select * from "{table}"'))
   ...:
Out[2]:
^\[(.+)\] \[error\] \[client (.+)\] Directory index forbidden by rule: (.+)$
┌──────────────────────────┬─────────────────┬────────────────┐
│            c0            │       c1        │       c2       │
│         varchar          │     varchar     │    varchar     │
├──────────────────────────┼─────────────────┼────────────────┤
│ Sun Dec 04 05:15:09 2005 │ 222.166.160.184 │ /var/www/html/ │
│ Sun Dec 04 07:45:45 2005 │ 63.13.186.196   │ /var/www/html/ │
│ Sun Dec 04 08:54:17 2005 │ 147.31.138.75   │ /var/www/html/ │
│ Sun Dec 04 09:35:12 2005 │ 207.203.80.15   │ /var/www/html/ │
│ Sun Dec 04 10:53:30 2005 │ 218.76.139.20   │ /var/www/html/ │
│ Sun Dec 04 11:11:07 2005 │ 24.147.151.74   │ /var/www/html/ │
│ Sun Dec 04 11:33:18 2005 │ 211.141.93.88   │ /var/www/html/ │
│ Sun Dec 04 11:42:43 2005 │ 216.127.124.16  │ /var/www/html/ │
│ Sun Dec 04 12:33:13 2005 │ 208.51.151.210  │ /var/www/html/ │
│ Sun Dec 04 13:32:32 2005 │ 65.68.235.27    │ /var/www/html/ │
│            ·             │      ·          │       ·        │
│            ·             │      ·          │       ·        │
│            ·             │      ·          │       ·        │
│ Mon Dec 05 06:36:59 2005 │ 221.232.178.24  │ /var/www/html/ │
│ Mon Dec 05 09:09:48 2005 │ 207.12.15.211   │ /var/www/html/ │
│ Mon Dec 05 10:26:39 2005 │ 141.153.150.164 │ /var/www/html/ │
│ Mon Dec 05 10:28:44 2005 │ 198.232.168.9   │ /var/www/html/ │
│ Mon Dec 05 10:48:48 2005 │ 67.166.248.235  │ /var/www/html/ │
│ Mon Dec 05 14:11:43 2005 │ 141.154.18.244  │ /var/www/html/ │
│ Mon Dec 05 16:45:04 2005 │ 216.216.185.130 │ /var/www/html/ │
│ Mon Dec 05 17:31:39 2005 │ 218.75.106.250  │ /var/www/html/ │
│ Mon Dec 05 19:00:56 2005 │ 68.228.3.15     │ /var/www/html/ │
│ Mon Dec 05 19:14:09 2005 │ 61.220.139.68   │ /var/www/html/ │
├──────────────────────────┴─────────────────┴────────────────┤
│ 32 rows (20 shown)                                3 columns │
└─────────────────────────────────────────────────────────────┘

^\[(.+)\] \[error\] jk2_init\(\) Can't find child (\d+) in scoreboard$
┌──────────────────────────┬─────────┐
│            c0            │   c1    │
│         varchar          │ varchar │
├──────────────────────────┼─────────┤
│ Sun Dec 04 17:43:08 2005 │ 1566    │
│ Sun Dec 04 17:43:08 2005 │ 1567    │
│ Sun Dec 04 20:47:16 2005 │ 2082    │
│ Sun Dec 04 20:47:17 2005 │ 2085    │
│ Sun Dec 04 20:47:17 2005 │ 2086    │
│ Sun Dec 04 20:47:17 2005 │ 2087    │
│ Mon Dec 05 07:57:02 2005 │ 5053    │
│ Mon Dec 05 07:57:02 2005 │ 5054    │
│ Mon Dec 05 11:06:52 2005 │ 5619    │
│ Mon Dec 05 11:06:52 2005 │ 5620    │
│ Mon Dec 05 11:06:52 2005 │ 5621    │
│ Mon Dec 05 11:06:52 2005 │ 5622    │
├──────────────────────────┴─────────┤
│ 12 rows                  2 columns │
└────────────────────────────────────┘

^\[(.+)\] \[notice\] jk2_init\(\) Found child (\d+) in scoreboard slot (\d+)$
┌──────────────────────────┬─────────┬─────────┐
│            c0            │   c1    │   c2    │
│         varchar          │ varchar │ varchar │
├──────────────────────────┼─────────┼─────────┤
│ Sun Dec 04 04:51:08 2005 │ 6725    │ 10      │
│ Sun Dec 04 04:51:09 2005 │ 6726    │ 8       │
│ Sun Dec 04 04:51:09 2005 │ 6728    │ 6       │
│ Sun Dec 04 04:51:37 2005 │ 6736    │ 10      │
│ Sun Dec 04 04:51:38 2005 │ 6733    │ 7       │
│ Sun Dec 04 04:51:38 2005 │ 6734    │ 9       │
│ Sun Dec 04 04:52:04 2005 │ 6738    │ 6       │
│ Sun Dec 04 04:52:04 2005 │ 6741    │ 9       │
│ Sun Dec 04 04:52:05 2005 │ 6740    │ 7       │
│ Sun Dec 04 04:52:05 2005 │ 6737    │ 8       │
│            ·             │  ·      │ ·       │
│            ·             │  ·      │ ·       │
│            ·             │  ·      │ ·       │
│ Mon Dec 05 18:50:30 2005 │ 6733    │ 8       │
│ Mon Dec 05 18:56:03 2005 │ 6740    │ 7       │
│ Mon Dec 05 18:56:03 2005 │ 6741    │ 8       │
│ Mon Dec 05 19:00:43 2005 │ 6750    │ 8       │
│ Mon Dec 05 19:00:43 2005 │ 6749    │ 7       │
│ Mon Dec 05 19:00:54 2005 │ 6751    │ 10      │
│ Mon Dec 05 19:11:00 2005 │ 6780    │ 7       │
│ Mon Dec 05 19:14:08 2005 │ 6784    │ 8       │
│ Mon Dec 05 19:15:55 2005 │ 6791    │ 8       │
│ Mon Dec 05 19:15:55 2005 │ 6790    │ 7       │
├──────────────────────────┴─────────┴─────────┤
│ 836 rows (20 shown)                3 columns │
└──────────────────────────────────────────────┘
```

DuckDB also supports saving results to various output types such as CSV, JSON, and Parquet, among others. For more information, visit DuckDB's [documentation](http://duckdb.org/docs/archive/0.9.0/guides/python/export_pandas).

## How does it work?
```
┌───────────┐
│ Log Files │
└─┬─────────┘
  │
┌─▼───────────┐     ┌──────────────────────┐
│ Regex Sieve │─────► Text Embedding Model │
└─┬─────────▲─┘     └──────────────────────┘
  │         +         +
  │         +         +
┌─▼──────┐  +       ┌─▼─────┐
│ DuckDB │  + + + + │ GPT-4 │
└─┬──────┘          └───────┘
  │
┌─▼──────────────────────────┐
│ CSV / JSON / Parquet / ... │
└────────────────────────────┘


────>  log flow
+ + >  pattern flow
```

Log2Row processes all logs through a list of regex patterns. If a log matches a pattern successfully, it's grouped and variables are inserted into DuckDB. If a log doesn't match any patterns, it's buffered. These buffered logs are used to detect log communities via a text embedding model. Samples from each community are then sent to GPT-4 to extract their regex patterns.

The pattern flow isn't part of the main processing, which means that after an initial bootstrap, the processing speed increases significantly. Thus, the longer Log2Row runs, the higher the logs/sec rate it has.

## How to install it?
```
  ↳pip install git@https://github.com/ethe/log2row.git

  ↳ python3 -m log2row
Usage: python -m log2row [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  clean  log2row cache all extracted patterns to each files as default,...
  run
```

*Logs2Array requires Python3.9+*

## How much does it cost?
Log2Row uses GPT-4 to extract the regex of log community, each extraction would costs hundreds to thousands tokens of GPT-4. This means each log community detection would costs 0.01$ to 0.1$.

## What is next?
- [ ] auto-detect multiple parts of log templates
  ```
  I, [2023-09-13T00:00:04.832375 #6]  INFO -- : [a8e9e534-bf8b-42df-8ade-dd3f84af5bf0] Started GET "/" for 172.31.6.222 at 2023-09-13 00:00:04 +0000
  ▲                                                                                  ▲ ▲                                                           ▲
  ├──────────────────────────────────────────────────────────────────────────────────┘ ├───────────────────────────────────────────────────────────┘
  │^(I), \[([\d\-T:.]+) #(\d+)\]\s+(INFO) -- : \[([a-f0-9\-]+)\]$                      │^(Started) (GET) ("\/") for (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}) at ([\d\- :+]+)$
  ```
- [ ] more steps by GPT
  - [ ] type columns of each pattern
  - [ ] parse datetime
- [ ] not just a command-line tool but a filebeat-like sidecar component to streaming consume logs
- [ ] GPT-3.5 compatible

## More information
Currently, log2row is still in early stage. If you are interested in it, let's discuss it on [Hacker News](https://news.ycombinator.com/item?id=37789903)
