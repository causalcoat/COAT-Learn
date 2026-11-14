# Example data

Two CSV files support the three discovery examples. Row order and duplicate texts are preserved.

| File | Rows | Columns used |
| --- | ---: | --- |
| [Apple graph1](<apple_reviews_PAG_graph1.csv>) | 1,000 | MB: `Score`, `Review`; PAG: `Taste`, `Score`, `Review` |
| [Hillstrom posts](<hillstrom_email_visit_post.csv>) | 2,000 | ADJ: `email_type`, `visit`, `post` |

The apple file is an unchanged export of the authors' graph1 research data. Columns not named by the task do not enter discovery. Its SHA-256 is `3c9dd7a9d5b0177ad21fc83b460142e21d441c204ec516cfa44a7d42dc4c3b0d`.

The Hillstrom file contains synthetic customer posts and email/visit records from the authors' fixed 2,000-row example. It is not the original randomized Hillstrom customer table. `Yes` in `email_type` means sent; an empty cell means not sent. `visit` uses `Yes` and `No`. The ADJ configuration explicitly maps these values to 1 and 0. Its SHA-256 is `8ff3d819c0024f83aefe02892b531972c4da2adbab5b64aaf503a48b15c939bb`.

## License and attribution

Both CSV files are provided by the COAT authors under [Creative Commons Attribution 4.0 International (CC BY 4.0)](<LICENSE>). The software's Apache-2.0 license does not replace this data license. These files are supplied as-is, without warranties.

For attribution, credit Chenxi Liu, Yongqiang Chen, Tongliang Liu, Mingming Gong, James Cheng, Bo Han and Kun Zhang; name the COAT-Learn example dataset used; link to [this source directory](https://github.com/causalcoat/COAT-Learn/tree/main/data) and the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/); and indicate any modifications. Both CSV exports are unchanged in this publication. Related research: [JMLR 27(229):1–95, 2026](https://www.jmlr.org/papers/v27/25-2137.html).

## 中文

两份数据支持三个发现示例。MB 与 PAG 共用 1000 行苹果 graph1，分别读取 `Score/Review` 和 `Taste/Score/Review`；ADJ 读取 2000 行合成顾客帖子中的 `email_type/visit/post`。程序只使用配置指定的列，保留行序和重复文本。

Hillstrom 示例不是原始随机化顾客表。邮件列的 `Yes` 表示已发送，空值表示未发送；访问列的 `Yes/No` 表示是否访问。配置明确记录二元编码映射。两份 CSV 均由 COAT 作者以 [CC BY 4.0](<LICENSE>)发布，与软件的 Apache-2.0 许可分开。再分发时注明上述作者、所用数据集名称、来源链接与许可，并说明修改。本次发布未修改 CSV 内容，数据按原样提供，不作保证。
