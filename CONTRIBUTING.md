# Contributing to Notebooks Extended 
Contributions to notebooks-extended fall into the following 4 categories.
### 1. Find bugs in existing notebooks:
- Sometimes things unexpectedly break in our notebooks
    - File an [issue](https://github.com/rapidsai/notebooks-extended/issues/new/choose)
describing what you encountered or what you want to see changed
    - The RAPIDS team will evaluate the issues and triage them, scheduling
them for a release
        - If you believe the issue needs priority attention, please
comment on the issue to notify the team
### 2. Implement a bug fix or satisfy a feature request
- To implement a feature or bug-fix for an existing outstanding issue, please 
    follow the [code contributions](#code-contributions) guide below
    - If you need more context on a particular issue, please ask in a comment
### 3. Propose and create a new notebook:
- To propose and implement a new notebook, please file a new feature request 
    [issue](https://github.com/rapidsai/notebooks-extended/issues/new/choose)
    - Describe the intended feature and discuss the design & implementation with the community
    - Once the team agrees that the plan looks good, implement it using the [code contributions](#code-contributions) guide below
    - Test your notebook using our [testing guide](TESTING.md)!
    - Be sure that your PR includes an update the notebooks-contrib REAMDE with your notebook(s)'s title, location, description, and link.  Failure to do so will delay your PR's merge until completion.
### 4. Create a tutorial or code walk through:
- Show your expertise in RAPIDS while teaching people how to use RAPIDS in their data science pipeline

## Code contributions

### Your first issue

1. Follow the guide at the bottom of this page for [Setting Up Your Build Environment](#setting-up-your-build-environment).
2. Find an issue to work on. The best way is to look for the [good first issue](https://github.com/rapidsai/notebooks-extended/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
    or [help wanted](https://github.com/rapidsai/notebooks-extended/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) labels.
3. Comment on the issue stating that you are going to work on it.
4. Code! Make sure to update unit tests!
5. When done, [create your pull request](https://github.com/rapidsai/notebooks-extended/compare).
6. Verify that CI passes all [status checks](https://help.github.com/articles/about-status-checks/). Fix if needed.
7. Wait for other developers to review your code and update code as needed.
8. Once reviewed and approved, a RAPIDS developer will merge your pull request.

Remember, if you are unsure about anything, don't hesitate to comment on issues
and ask for clarifications!

### Seasoned developers

Once you have gotten your feet wet and are more comfortable with the code, you
can look at the prioritized issues for our next release in our [project boards](https://github.com/rapidsai/notebooks-extended/projects).

> **Pro Tip:** Always look at the release board with the highest number for
issues to work on. This is where RAPIDS developers also focus their efforts.

Look at the unassigned issues, and find an issue to which you are comfortable
contributing. Start with _Step 3_ above, commenting on the issue to let
others know you are working on it. If you have any questions related to the
implementation of the issue, ask them in the issue instead of the PR.

## Setting Up Your Build Environment

The following instructions are for developers and contributors to cuDF OSS development. These instructions are tested on Linux Ubuntu 16.04 & 18.04. Use these instructions to build cuDF from source and contribute to its development.  Other operating systems may be compatible, but are not currently tested.
