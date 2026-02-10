# LFX Mentorship Cover Letter
## Kubeflow Headlamp Plugin Integration Project

**Linux Foundation Mentorship Program – Spring 2026**  
**Project ID:** [https://mentorship.lfx.linuxfoundation.org/project/abe20383-80fd-496c-8a5a-453fcb732f55](https://mentorship.lfx.linuxfoundation.org/project/abe20383-80fd-496c-8a5a-453fcb732f55)  
**GitHub Issue:** [https://github.com/kubernetes-sigs/headlamp/issues/3710](https://github.com/kubernetes-sigs/headlamp/issues/3710)

---

## Applicant Information

**Name:** Prachi Agrawal  
**Institution:** ABV-Indian Institute of Information Technology and Management (IIITM), Gwalior, India  
**Program:** B.Tech (Information Technology) + MBA (Dual Degree)  
**Current Year:** 3rd Year (2023–2028)  
**Location:** Gwalior, Madhya Pradesh, India  
**Timezone:** Indian Standard Time (IST, GMT+5:30)

### Contact Information
- **Email:** [agrawalprachi7718@gmail.com](mailto:agrawalprachi7718@gmail.com)
- **Phone:** +91 7489401140
- **GitHub:** [https://github.com/Prachi194agrawal](https://github.com/Prachi194agrawal) (71 public repositories)
- **LinkedIn:** [https://linkedin.com/in/prachi-agrawal](https://linkedin.com/in/prachi-agrawal)
- **Codeforces:** Expert (Rating: 1604)
- **LeetCode:** 700+ problems solved
- **Kaggle:** Top 872/10,000 (Amazon ML Challenge)

---

## 1. How Did I Find Out About This Mentorship Program?

I discovered the Linux Foundation Mentorship (LFX) program while exploring cloud-native open-source contribution opportunities after becoming a CNCF KubeEdge contributor and getting recognized with Green Channel status for consistent, high-quality contributions. While looking for projects that combine Kubernetes, frontend development, and MLOps, I found the Headlamp ecosystem and the Kubeflow plugin project via the Kubernetes Slack #headlamp channel and the LFX mentorship portal.

The Kubeflow Headlamp Plugin Integration project immediately resonated with me because it formalizes a problem I have personally faced: **fragmented observability between Kubernetes cluster resources and ML workflows**. In my own projects, I have often had to juggle between Kubernetes dashboards, ML UIs, and metrics tools to debug issues, which is exactly the gap this mentorship aims to close.

---

## 2. Why Am I Interested in This Program?

### Personal Experience With the Problem

In my **Cross-Camera Player Mapping System**, I ran distributed YOLOv8-based inference across multiple camera streams, and in the **AI Career Coach Platform**, I orchestrated background ML workflows with Inngest on serverless infrastructure. Debugging issues in these systems often required switching between logs, dashboards, and monitoring UIs. I repeatedly felt the need for a single pane of glass where I could see:

- Which ML jobs are running, failing, or stuck
- Resource consumption of ML workloads in the cluster
- Clear links from "this job is failing" to "here are its pods, events, and metrics"

When I read the Kubeflow + Headlamp proposal, it felt like a direct, general solution to the exact pain points I had already encountered in my own work.

### Alignment With My Long-Term Goals

I am intentionally building my career at the intersection of:
- MLOps and scalable ML systems
- Kubernetes and cloud-native tools
- Developer / operator experience (DX/Ops UX)

From my background: I specialize in Software Engineering and Scalable Systems, have built production-ready applications, and have contributed to CNCF KubeEdge in the healthcare ML domain. This LFX project is a perfect fit: it is about building a production-grade integration that makes ML workflows visible and manageable directly inside a Kubernetes UI.

### Impact on the Cloud-Native ML Community

Kubeflow users and Kubernetes operators repeatedly ask for better integration between Kubeflow's ML abstractions and Kubernetes-native observability tools. This plugin directly addresses those needs by:

- Showing Kubeflow CRDs (Pipelines, Notebooks, Katib Experiments) in Headlamp
- Providing real-time status updates and dashboards
- Linking both ways between Kubeflow UIs and Headlamp views

I see this project as a way to help not just myself, but many teams who are running ML workloads on shared Kubernetes clusters and struggling with fragmented tooling.

---

## 3. What Experience and Knowledge/Skills Do I Have That Are Applicable?

### 3.1 Machine Learning & MLOps Domain Understanding

From my projects and competitions:

#### Cross-Camera Player Mapping System (Dec 2024)
- Implemented a YOLOv8-based re-identification system using global assignment optimization across multiple unsynchronized camera feeds
- Achieved 95% tracking accuracy across 4+ simultaneous camera streams in real-time
- **Relevance:** Understanding distributed ML inference, latency, throughput, and monitoring needs for such systems in a cluster

#### AI Career Coach Platform (Mar 2025)
- Architected a full-stack Next.js application integrated with OpenAI APIs and automated background workflows via Inngest
- Deployed on Vercel with a scalable serverless architecture to handle concurrent users and asynchronous jobs
- **Relevance:** Real-time data flows, background job orchestration, and the need for clear visibility into job states and failures

#### Kaggle / Amazon ML Challenge
- Worked on large-scale ML pipelines for real-estate price prediction (Gurgaon dataset with 500K+ records)
- Ranked among the top 10% of 20,000+ participants in Amazon ML Challenge (Top 872/10,000)
- **Relevance:** Familiarity with ML pipelines, metrics (RMSE, MAE, precision, recall, F1), and performance monitoring at scale

#### Additional ML Projects
- **PCOS Detection Using Ultrasound Images:** YOLOv8-based medical imaging classification achieving 95% accuracy
- **Gender-Bias Summarizer:** BERT/RoBERTa transformer models for NLP-based bias detection

**This background means I am not just rendering CRDs in a UI; I understand what ML engineers and operators actually care about when looking at runs, experiments, and notebooks.**

### 3.2 Full-Stack & Frontend Skills (React / TypeScript)

My experience in full-stack development with modern web technologies:

**Languages:** JavaScript, TypeScript, HTML/CSS  
**Frameworks:** React.js, Next.js, Node.js, Express.js  
**APIs & Auth:** RESTful APIs, Clerk Auth  
**Styling:** Tailwind CSS, modern CSS frameworks  
**Hosting:** Vercel, AWS (EC2)

For this project, those skills map naturally to:
- Implementing Headlamp plugin UIs with React + TypeScript
- Designing list and detail views with filters, sorting, search
- Handling real-time updates using Kubernetes watch APIs or polling
- Ensuring responsive, accessible UI (keyboard navigation, ARIA labels, consistent styling)

The AI Career Coach Platform in particular proves I can build and deploy complex frontends backed by asynchronous workflows and third-party APIs, which is structurally similar to consuming Kubernetes and Kubeflow APIs in a plugin.

### 3.3 Kubernetes, Cloud, and CNCF Experience

From my open-source contributions and DevOps experience:

#### CNCF KubeEdge Contributor (Green Channel)
- Designed and implemented high-performance data processing modules for healthcare ML workloads at the edge
- Learned CNCF-level code quality standards, community review workflows, and performance-aware design
- **Green Channel Status:** Recognized for consistent, high-quality contributions

#### DevOps / Tools
- **Proficient in:** Git, GitHub, Docker, AWS (EC2), Linux/Unix, Vercel, Inngest
- Comfortable working in cloud-native environments, containers, and CI/CD-driven workflows
- KubeEdge contributions show that I already understand how ML and Kubernetes intersect in real scenarios and can work under the expectations of a CNCF project

### 3.4 Algorithmic & Engineering Rigor

- **Codeforces Expert (Rating 1604)**
- **Top 100 in Yandex Cup (Software Track)**
- **700+ problems on LeetCode**

These demonstrate that I can reason about complexity, optimize performance, and write clean, efficient code, which is crucial when working with real-time dashboards and large resource lists in Headlamp.

---

## 4. What Do I Hope to Get Out of This Mentorship?

### 4.1 Technical Growth

- **Deep understanding of Headlamp's plugin system:** How to use `registerSidebarEntry`, `registerDetailsViewSection`, and related APIs to extend the UI safely and idiomatically
- **Hands-on experience with Kubernetes watch APIs:** Designing robust real-time update flows for CRDs and handling reconnection, backoff, and edge cases
- **End-to-end MLOps observability patterns:** Learning how best to represent Kubeflow resources, metrics, and events in a way that is useful to both operators and ML engineers

### 4.2 Professional & Community Development

- Work closely with maintainers like illume and other Headlamp contributors, learning how they review, design, and stabilize production plugins
- Improve my skills in technical communication: writing design docs, explaining trade-offs, and handling feedback in public GitHub discussions and Kubernetes Slack
- Build a public, production-quality artifact (the plugin) that I can maintain and evolve over time, not just during the mentorship

### 4.3 Personal Fulfillment

This project is almost exactly at the intersection of the three things I care about most right now:
- Machine learning workflows and their practical challenges
- Kubernetes and cloud-native infrastructure
- Building tools that make other developers' and operators' lives easier

Contributing to this plugin will be both a learning experience and a meaningful way to give back to the community that powers so much of modern ML infrastructure.

---

## 5. Understanding the Kubeflow–Headlamp Problem (In My Own Words)

From the issue and project brief, the core idea is:

**The Problem:**
- Kubeflow exposes rich ML resources via CRDs: Pipelines, Experiments, Runs, Notebooks, Katib Experiments, Trainer jobs, SparkApplications, etc.
- Headlamp is an operator-friendly Kubernetes UI but currently treats these CRDs as opaque objects, without tailored views or ML-aware context
- Operators must context-switch between Headlamp, Kubeflow's own UIs, and possibly Grafana/Prometheus dashboards to understand and debug workflows

**The Solution (Plugin Goals):**
1. Add Kubeflow as a first-class "section" in Headlamp with sidebar entries for ML resources
2. Provide list + detail views for Kubeflow CRDs with the right ML-focused fields (status, start/end time, metrics, owner, experiment, etc.)
3. Build an "Active & Failed Runs" dashboard that surfaces high-signal information quickly
4. Integrate events and metrics, ideally tying Kubernetes events and Prometheus metrics back into the ML context
5. Provide one-click deep links from Headlamp to Kubeflow UIs and, optionally, from Kubeflow to Headlamp for debugging underlying Kubernetes resources

**This is not about replacing Kubeflow's UI, but about embedding Kubeflow awareness into the operator's main Kubernetes UI and wiring in convenient handoffs.**

---

## 6. High-Level Technical Approach

I plan to follow the outlined phases in the issue, aligning with the given timeline, while using my stack:

**Technology Stack:**
- **Frontend:** React.js + TypeScript
- **Data:** Kubernetes API via client libraries (e.g., @kubernetes/client-node)
- **Real-time:** Watch APIs and/or polling, depending on what is most robust in the Headlamp plugin environment
- **UI:** Headlamp's plugin APIs, consistent design system, and accessible React components

### Key Implementation Elements

#### 1. Kubeflow Sidebar Section in Headlamp
- "Kubeflow" parent entry with children: "Pipelines", "Notebooks", "Katib Experiments", "Overview"
- Each entry routes to a plugin route (`/kubeflow/pipelines`, etc.) with React components rendering lists/dashboards

#### 2. List and Detail Views
- Tables for each CRD type with namespace filters, search, and sorting by status/time
- Detail pages that show: status, metadata, events, owner references, and deep links to Kubeflow UIs

#### 3. Active & Failed Runs Dashboard
- Overview page aggregating active and recently failed runs across namespaces
- Filters by namespace, type, and optionally owner
- Summaries of "what needs attention right now"

#### 4. Real-Time Updates and Events
- Use Kubernetes watches to update run statuses live
- Attach events (Warnings/Normals) to specific resources to explain state changes (e.g., OOMKilled)

#### 5. Metrics Integration (If Time Permits)
- Wire in Prometheus metrics if cluster has it, to show resource usage/metrics for Kubeflow jobs (GPU/CPU utilization, runtime)

#### 6. Links Between UIs
- From Headlamp → Kubeflow: buttons like "Open in Kubeflow Pipelines UI"
- Potentially from Kubeflow → Headlamp: using in-cluster app links to jump to Headlamp for underlying Kubernetes resources

---

## 7. Project Timeline (12 Weeks, Adapted to LFX)

I will align with the issue's proposed phases while mapping them to concrete weekly deliverables.

### Phase 1 (Weeks 1–2): Design & Setup
- Set up local cluster with Kubeflow and Headlamp
- Study existing Headlamp plugins (Karpenter, Volcano, Spark Operator) to replicate patterns
- Finalize which Kubeflow CRDs to support first (Pipelines, Notebooks, Katib Experiments)
- Prepare UX mockups for:
  - Sidebar structure
  - List views
  - Detail views
  - Overview dashboard
- Share design doc + mockups with mentors on GitHub and Slack for feedback

### Phase 2 (Weeks 3–5): Basic Views (MVP)
**Implement:**
- Sidebar entries and routing for Kubeflow section
- List views for pipeline runs, notebooks, Katib experiments
- Detail views with basic metadata (name, namespace, status, timestamps) and links to Kubeflow UIs
- Integrate with Headlamp navigation so Kubeflow resources are visible and navigable inside Headlamp

**Deliverable:** Ship Alpha 1 - minimal but functional plugin for early feedback

### Phase 3 (Weeks 6–8): Live Updates & Dashboard
- Add watch-based or polling-based live updates for resource statuses
- Implement an "Active & Failed Runs" dashboard showing:
  - Currently running pipelines/experiments
  - Recently failed runs
  - Long-running jobs with durations
- Add Kubernetes events to detail pages for context and debugging help
- If time and environment allow, integrate Prometheus metrics (resource usage graphs, basic metrics)

### Phase 4 (Weeks 9–10): Enhanced Features & Polish
- Add support for editing Kubeflow CRDs via Headlamp's YAML editor where it makes sense (e.g., modifying Notebook specs or retriggering runs)
- Improve UI polish: icons, badges, better filtering, and performance tuning for large sets of resources
- Ensure robust behavior when:
  - Some Kubeflow components are not installed
  - UIs or services are temporarily unavailable

### Phase 5 (Weeks 11–12): Testing & Documentation
**Comprehensive testing across:**
- Different Kubeflow setups
- Permissions (RBAC)
- Browser environments

**Finalize:**
- README and user guide
- Developer guide on how plugin is structured and can be extended
- Screenshots and/or demo video illustrating key flows
- Prepare for community review and final evaluation

---

## 8. Why I Choose Kubeflow Over Other Headlamp LFX Projects

Given the available Headlamp-related LFX projects (Cluster API, Knative, Strimzi, Volcano, Kubeflow), I prioritize Kubeflow because:

1. **Direct ML workflow visibility**, which aligns with my ML + systems background
2. **Strongly UI-centric** and rich in operator/ML user interactions, which fits my frontend + full-stack strengths
3. **Addresses a pain point I have personally experienced** while running ML workloads

Other projects like Cluster API and Strimzi are more infrastructure-focused and less centered on ML workflows; they are interesting, but Kubeflow is clearly the best match for my current skills and long-term goals.

---

## 9. Open-Source Pull Requests & Contributions

As requested in the project description, here are examples of my open-source contributions demonstrating code quality, communication skills, and ability to work with maintainers in public repositories.

### 1. Healthcare ML Data Processing Contributions - CNCF KubeEdge

**Repository:** kubeedge/kubeedge  
**Status:** Active Contributor (Green Channel Status ✅)  
**Contribution Period:** 2024–Present

**Problem Solved:**  
Healthcare ML workloads at the edge lacked standardized, high-performance data processing modules, making it difficult to deploy and benchmark medical imaging models (DICOM, NIfTI formats) on resource-constrained edge devices like Raspberry Pi and Jetson Nano.

**Technical Approach:**
- Designed Python-based data ingestion and processing pipeline supporting 5+ medical image formats
- Implemented parallel processing optimizations for edge devices with limited compute resources
- Followed CNCF code quality standards (linting, formatting, comprehensive documentation)
- Created benchmark scripts to measure performance across different edge hardware configurations
- Integrated with KubeEdge's edge-cloud messaging architecture for seamless data flow

**Communication & Community Engagement:**
- Participated actively in KubeEdge community meetings and design discussions
- Responded promptly to maintainer feedback and code review suggestions
- Maintained consistent code quality and documentation standards
- Collaborated with other contributors on healthcare ML use cases
- **Recognized with Green Channel contributor status** for reliability and consistent high-quality contributions

**Impact:**
- Earned Green Channel contributor status from CNCF KubeEdge maintainers
- Contributions enable healthcare ML research teams to deploy models on edge infrastructure
- Improved data processing performance on resource-constrained devices
- Strengthened KubeEdge's position in the healthcare edge computing ecosystem

---

### 2. [Your Second Best PR - Add Here]

**Repository:** [Organization/Repo Name]  
**PR Link:** [Direct GitHub PR URL]  
**Status:** Merged ✅ / Open / In Progress

**Problem Solved:**  
[2-3 sentences describing what issue this PR addressed and why it was important]

**Technical Approach:**
- [Key technical decision 1]
- [Key technical decision 2]
- [Technologies/frameworks used]
- [Tests added, performance improvements, etc.]

**Communication & Review Process:**
- [How you wrote the PR description]
- [How you responded to code review feedback]
- [Number of review iterations]
- [Any design discussions or community feedback]

**Impact:**
- [What happened after the PR was merged]
- [Usage stats or recognition, if available]
- [Follow-up contributions or community response]

---

### 3. [Your Third Best PR - Add Here]

**Repository:** [Organization/Repo Name]  
**PR Link:** [Direct GitHub PR URL]  
**Status:** Merged ✅ / Open / In Progress

**Problem Solved:**  
[2-3 sentences describing the problem]

**Technical Approach:**
- [Key technical details]
- [Implementation choices]
- [Testing strategy]

**Communication & Review Process:**
- [Your approach to collaboration]
- [Feedback handling]
- [Community interaction]

**Impact:**
- [Results and recognition]
- [Community benefit]

---

### 4. Additional Open-Source Projects (GitHub Profile)

**Public Repositories:** 71 repositories demonstrating consistent coding practices  
**GitHub Profile:** [https://github.com/Prachi194agrawal](https://github.com/Prachi194agrawal)

**Notable Projects with Professional Git Practices:**
- **Cross-Camera Player Mapping System:** Clean commit history, detailed README, modular code structure
- **AI Career Coach Platform:** Feature branches, semantic commits, comprehensive documentation
- **PCOS Detection System:** Medical imaging ML pipeline with reproducible results
- **Gurgaon Real Estate Price Prediction:** Data science project with Jupyter notebooks and visualization dashboards

All projects follow professional development practices:
- Clear commit messages
- Issue tracking and project boards
- Comprehensive README files with setup instructions
- Modular, well-documented code
- Test coverage where applicable

---

**Note:** _I am actively working on creating PRs for the Headlamp project during the application period to demonstrate my understanding of the codebase and plugin architecture. These contributions will further showcase my ability to work within the Kubernetes SIG-UI ecosystem._

---

## 10. Commitment & Post-Mentorship Plans

### Time & Availability

I can commit **25–30 hours per week** during the 12-week mentorship period. My academic schedule at ABV-IIITM Gwalior allows flexibility during the mentorship period, and I have prior experience balancing intensive coding commitments with academics (evidenced by maintaining Codeforces Expert rating, 700+ LeetCode problems, and multiple project deliveries while pursuing my dual degree program).

### Communication

I commit to:
- **Weekly synchronization meetings** (flexible timing around IST, typically evenings work best)
- **Active presence on GitHub issues** and Kubernetes Slack (#headlamp channel)
- **Bi-weekly written progress reports** summarizing:
  - Completed deliverables
  - Code PRs with links
  - Blockers or technical challenges
  - Next steps and timeline adjustments
- **Transparent communication** about any scheduling conflicts or challenges

### Post-Mentorship

I intend to:
- **Continue maintaining and improving** the Kubeflow–Headlamp plugin after the LFX program concludes
- **Respond to user issues** and help triage/resolve bugs reported by the community
- **Extend support** for additional Kubeflow components where they bring real operator value (e.g., SparkApplication, Training Operator)
- **Help onboard new contributors** interested in extending this plugin or creating similar Headlamp integrations
- **Document learnings** through blog posts and technical talks to help others understand MLOps observability patterns

---

## Conclusion

This LFX project sits exactly where my skills, experience, and interests intersect: **ML workflows, Kubernetes, and developer/operator tooling**. My background in competitive programming (Codeforces Expert), full-stack engineering (React/TypeScript/Next.js), CNCF KubeEdge contributions (Green Channel status), and ML projects (YOLOv8 systems, Kaggle Top 10%) has prepared me to handle both the UX and systems challenges of building a robust Kubeflow plugin for Headlamp.

I am excited about the opportunity to work closely with the Headlamp maintainers, learn from their experience, and ship a plugin that meaningfully improves how operators and ML engineers understand and manage their Kubeflow workloads inside Kubernetes.

Thank you for considering my application. I look forward to contributing to the Kubernetes ecosystem and growing as an engineer through this mentorship.

---

**Prachi Agrawal**  
Email: [agrawalprachi7718@gmail.com](mailto:agrawalprachi7718@gmail.com)  
GitHub: [https://github.com/Prachi194agrawal](https://github.com/Prachi194agrawal)  
LinkedIn: [https://linkedin.com/in/prachi-agrawal](https://linkedin.com/in/prachi-agrawal)  
Codeforces: [https://codeforces.com/profile/Prachi194agrawal](https://codeforces.com/profile/Prachi194agrawal)
made pdf