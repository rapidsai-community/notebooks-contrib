# Contributing to cuDF

Contributions to notebooks-contrib fall into the following 5 categories.

You can contribute to the notebooks-extended repo in 5 ways:
1. Finding bugs in existing notebooks:
    - Sometimes things unexpectedly break in our notebooks.
    - File an [issue](https://github.com/rapidsai/notebooks-extended/issues/new/choose)
    describing what you encountered or what you want to see changed.
    - The RAPIDS team will evaluate the issues and triage them, scheduling
    them for a release. If you believe the issue needs priority attention
    comment on the issue to notify the team.
    
    Please raise an issue so we can fix it!
    
2. Peer reviewing and benchmarking new and existing notebooks:
    - Both new and existing notebooks need to be checked against current and new RAPIDS library releases. 
    - Your help is truly appreciated in making sure that those notebooks not just work, but are efficient, effective and run rapidly as well.
3. Implement a bug fix or satsfy a feature request
    - To implement a feature or bug-fix for an existing outstanding issue, please 
    Follow the [code contributions](#code-contributions) guide below. If you 
    need more context on a particular issue, please ask in a comment.
4. Propose and create a new notebook:
    - To propose and implement a new Feature, please file a new feature request 
    [issue](https://github.com/rapidsai/notebooks-extended/issues/new/choose). Describe the 
    intended feature and discuss the design and implementation with the team and
    community. Once the team agrees that the plan looks good, go ahead and 
    implement it, using the [code contributions](#code-contributions) guide below.
5. Creating a tutoral or code walk through blog/youtube video/sharing your :
    - Show your expertise in RAPIDS while teaching people how to use RAPIDS in their data science pipeline.
1. To report a bug, request a new feature, or report a problem with
    documentation, please file an [issue](https://github.com/rapidsai/cudf/issues/new/choose)
    describing in detail the problem or new feature. The RAPIDS team evaluates 
    and triages issues, and schedules them for a release. If you believe the 
    issue needs priority attention, please comment on the issue to notify the 
    team.
2. To propose and implement a new Feature, please file a new feature request 
    [issue](https://github.com/rapidsai/cudf/issues/new/choose). Describe the 
    intended feature and discuss the design and implementation with the team and
    community. Once the team agrees that the plan looks good, go ahead and 
    implement it, using the [code contributions](#code-contributions) guide below.
3. To implement a feature or bug-fix for an existing outstanding issue, please 
    Follow the [code contributions](#code-contributions) guide below. If you 
    need more context on a particular issue, please ask in a comment.

## Code contributions

### Your first issue

1. Follow the guide at the bottom of this page for [Setting Up Your Build Environment](#setting-up-your-build-environment).
2. Find an issue to work on. The best way is to look for the [good first issue](https://github.com/rapidsai/cudf/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
    or [help wanted](https://github.com/rapidsai/cudf/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) labels.
3. Comment on the issue stating that you are going to work on it.
4. Code! Make sure to update unit tests!
5. When done, [create your pull request](https://github.com/rapidsai/cudf/compare).
6. Verify that CI passes all [status checks](https://help.github.com/articles/about-status-checks/). Fix if needed.
7. Wait for other developers to review your code and update code as needed.
8. Once reviewed and approved, a RAPIDS developer will merge your pull request.

Remember, if you are unsure about anything, don't hesitate to comment on issues
and ask for clarifications!

### Seasoned developers

Once you have gotten your feet wet and are more comfortable with the code, you
can look at the prioritized issues for our next release in our [project boards](https://github.com/rapidsai/cudf/projects).

> **Pro Tip:** Always look at the release board with the highest number for
issues to work on. This is where RAPIDS developers also focus their efforts.

Look at the unassigned issues, and find an issue to which you are comfortable
contributing. Start with _Step 3_ above, commenting on the issue to let
others know you are working on it. If you have any questions related to the
implementation of the issue, ask them in the issue instead of the PR.

## Setting Up Your Build Environment

The following instructions are for developers and contributors to cuDF OSS development. These instructions are tested on Linux Ubuntu 16.04 & 18.04. Use these instructions to build cuDF from source and contribute to its development.  Other operating systems may be compatible, but are not currently tested.

### Code Formatting

TODO with the [new marketing branding](https://rapids.ai/branding.html)
