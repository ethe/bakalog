Ever found yourself in the same predicament as I have? As a backend developer, I often find myself in a dilemma during service program diagnostics: I regret not having collected and structured more logs in advance for insertion into Elasticsearch, while also being panic by the complexity of grep, awk and sed commands.

Recently, however, I seem to have seen a turning point: I've made a new attempt based on some recent innovations. I've created a command-line tool that leverages the LLM's automatic structuring capabilities to structure logs after the fact, and uses an in-process localized OLAP database, Python REPL, and Numpy / Pandas to provide a quick and powerful querying and processing workstation. You can [check out the results here](https://github.com/ethe/bakalog). Below are some of my thoughts on the issue of log processing.

## STRUCTURED LOGS ARE VALUABLE

Service applications typically output logs in text format. Developers can easily browse filtered logs based on text search and processing, but it's challenging to derive valuable insights: how long does each user stay on each page on average? Did the distribution of page response times change before a failure? All of these depend on structuring the logs and extracting the data for computation and aggregation, which is the premise for complex log processing and analysis.

## TWO WAYS OF STRUCTURING LOGS: PREPROCESSING AND POST-PROCESSING

Pre-structuring involves planning log patterns for easy handling, designing reasonable separators, and then using log collectors such as Filebeat and Vector to extract logs and write them into OLAP databases like ClickHouse and Hive, which have pre-set schemas. This is suitable for specific scenarios, such as user clickstreams and order records.

Post-structuring is by no means unfamiliar to operation engineers: a large amount of raw logs are casually discarded in disk files. Experienced operation engineers use incredible grep, awk, and sed magic commands to filter, transform, and analyze, panning for gold in the mud and sand.

## PRESTRUCTURING NEVER ENDS

Prestructuring is a painstaking tug of war. Prestructuring logs relies on prior organization, which means a lot of tedious and troublesome preparation work, because:

- A large number of different components will use the same channel to output logs: stdout, stderr, or output to the same log file, which is common in cloud-native environments;

- Even within the same process, and using a unified logger to output logs, there often exist multi-level structures, such as:

  - ```
    081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_38865049064139660 terminating
    081109 204005 35 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.73.220:50010 is added to blk_7128370237687728475 size 67108864
    081109 204106 329 INFO dfs.DataNode$PacketResponder: PacketResponder 2 for block blk_-6670958622368987959 terminating
    081109 204132 26 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.43.115:50010 is added to blk_3050920587428079149 size 67108864
    ```

  -  They have different log structures and fields:

  - ```
    081109 203615 <*> INFO dfs.DataNode$PacketResponder: PacketResponder <*> for block <*> terminating
    081109 204005 <*> INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: <*> is added to <*> size <*>
    ```

Such log types in the service may range from dozens to hundreds, which means we also need to write dozens or even hundreds of different rules to extract all types of logs. And most of these logs may only be used temporarily when the service exhibits abnormal behavior. Who would do so much preparation in advance for the rare temporary analysis and tracking logs? In the end, a large number of logs are not preprocessed, but are stored in files in the form of a "log lake."

## POST-STRUCTURING LACKS SUITABLE TOOLS

Every developer, more or less, uses grep, awk, and sed to quickly build a temporary analysis platform. Although they can achieve the effect of a powerful data analysis pipeline in the hands of masters, they are still too difficult for most developers:

- There is no reentrant single-step interactive interface for debugging;
- Matching and processing based on text is very tedious;
- There is a lack of ready-made analysis tool chains, and command-line tools are hard to deal with more complex queries and calculations.

## A BETTER AD-HOC LOG ANALYSIS TOOL

Is such a tool possible? It instantly structures existing text logs when I need it, provides rich data analysis capabilities, and does not require me to manage it in advance. We already have a mature data analysis workbench: Pandas combined with IPython or Jupyter. Python REPL provides a universal interactive interface, using Pandas makes Python REPL the best single-step log processing and analysis workbench.

And [DuckDB](https://duckdb.org/) is the potential best choice for data query and temporary table construction:

1. DuckDB can be easily locally deployed and linked into Python REPL to execute SQL queries;
2. Compared with SQLite, DuckDB is built on column storage, which can process data analysis calculations and queries faster;
3. DuckDB can export the results in Parquet format, which is convenient for transferring the preliminary processed data to a more professional data processing pipeline;
4. DuckDB can interact friendly with Numpy / Pandas.

Therefore, what we need is just a frontend that structures logs ad-hoc and writes into DuckDB. How to intelligently classify logs and detect the structure of logs used to be a difficult task, but now based on LLM and other AI derivative technologies, we can easily achieve this. For this, I tried to write the [bakalog](https://github.com/ethe/bakalog) to try to connect the whole process.

[Text Embedding](https://huggingface.co/blog/getting-started-with-embeddings) is the latest way to compare text similarity. It transforms sentences (such as logs) into fixed-length vectors. We can then compare the similarity between logs based on cosine similarity, thereby converting log text into different log clusters.

After clustering the logs, we need to identify the log patterns to facilitate the extraction of log variables, and further structure them:

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│   [Sun Dec 04 04:47:44 2005] [error] mod_jk child workerEnv in error state 6             │
│   [Sun Dec 04 04:51:08 2005] [notice] jk2_init() Found child 6725 in scoreboard slot 10  │
│   [Sun Dec 04 04:51:09 2005] [notice] jk2_init() Found child 6726 in scoreboard slot 8   │
│   [Sun Dec 04 04:51:18 2005] [error] mod_jk child workerEnv in error state 6             │
│   [Sun Dec 04 04:51:18 2005] [error] mod_jk child workerEnv in error state 6             │
│   [Sun Dec 04 04:51:18 2005] [error] mod_jk child workerEnv in error state 6             │
│   [Sun Dec 04 04:51:37 2005] [notice] jk2_init() Found child 6736 in scoreboard slot 10  │
│   [Sun Dec 04 04:51:38 2005] [notice] jk2_init() Found child 6733 in scoreboard slot 7   │
│   [Sun Dec 04 04:51:38 2005] [notice] jk2_init() Found child 6734 in scoreboard slot 9   │
│                                           ...                                            │
└────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                             │
                                             │
     ┌───────────────────────────────────────▼─────────────────────────────────────────┐
     │  ^\[(.+)\] \[error\] mod_jk child workerEnv in error state (\d+)$               │
     │  ^\[(.+)\] \[notice\] jk2_init\(\) Found child (\d+) in scoreboard slot (\d+)$  │
     └────────────────┬──────────────────────────────────────────┬─────────────────────┘
                      │                                          │
                      │                                          │
    ┌─────────────────▼────────┬─────────┐  ┌────────────────────▼─────┬─────────┬─────────┐
    │            c0            │   c1    │  │            c0            │   c1    │   c2    │
    │         varchar          │ varchar │  │         varchar          │ varchar │ varchar │
    ├──────────────────────────┼─────────┤  ├──────────────────────────┼─────────┼─────────┤
    │ Mon Dec 05 19:14:11 2005 │ 6       │  │ Mon Dec 05 19:15:55 2005 │ 6790    │ 7       │
    │ Mon Dec 05 19:11:04 2005 │ 6       │  │ Mon Dec 05 19:15:55 2005 │ 6791    │ 8       │
    │ Mon Dec 05 19:00:54 2005 │ 6       │  │ Mon Dec 05 19:14:08 2005 │ 6784    │ 8       │
    │ Mon Dec 05 19:00:44 2005 │ 6       │  │ Mon Dec 05 19:11:00 2005 │ 6780    │ 7       │
    │ Mon Dec 05 19:00:44 2005 │ 6       │  │ Mon Dec 05 19:00:54 2005 │ 6751    │ 10      │
    │ Mon Dec 05 18:56:04 2005 │ 6       │  │ Mon Dec 05 19:00:43 2005 │ 6749    │ 7       │
    │ Mon Dec 05 18:56:04 2005 │ 6       │  │ Mon Dec 05 19:00:43 2005 │ 6750    │ 8       │
    │ Mon Dec 05 18:50:31 2005 │ 6       │  │ Mon Dec 05 18:56:03 2005 │ 6741    │ 8       │
    │ Mon Dec 05 18:45:53 2005 │ 6       │  │ Mon Dec 05 18:56:03 2005 │ 6740    │ 7       │
    │ Mon Dec 05 18:45:53 2005 │ 6       │  │ Mon Dec 05 18:50:30 2005 │ 6733    │ 8       │
    │            ·             │ ·       │  │            ·             │  ·      │ ·       │
    │            ·             │ ·       │  │            ·             │  ·      │ ·       │
    │            ·             │ ·       │  │            ·             │  ·      │ ·       │
    └────────────────────────────────────┘  └──────────────────────────────────────────────┘
```

## WHAT'S NEXT?

This is just a Proof of Concept (PoC), it has more potential capabilities to grow into a real localized ad-hoc log analysis platform:

- Can a localized model be used to replace GPT-4?
- More steps? For example:
  - Automatically detect column types;
  - Parse datetime;
  - ...

If you like this idea, don't mind discussing its future with me.
