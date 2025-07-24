"""Test data generators for creating realistic test data."""

import random
from datetime import datetime, timedelta
from typing import Any


class RealisticTestDataGenerator:
    """Generates realistic test data for E2E tests."""

    # Realistic author names from various regions
    FIRST_NAMES = [
        "James",
        "Maria",
        "Wei",
        "Aisha",
        "Dmitri",
        "Priya",
        "Carlos",
        "Fatima",
        "Yuki",
        "Lars",
        "Sofia",
        "Ahmed",
        "Elena",
        "Rajesh",
        "Anna",
        "Chen",
        "Mohammed",
        "Julia",
        "Kenji",
        "Olga",
        "Samuel",
        "Mei",
        "Antonio",
        "Zara",
    ]

    LAST_NAMES = [
        "Smith",
        "García",
        "Zhang",
        "Patel",
        "Ivanov",
        "Johnson",
        "Williams",
        "Brown",
        "Jones",
        "Miller",
        "Davis",
        "Wilson",
        "Martinez",
        "Anderson",
        "Taylor",
        "Thomas",
        "Hernandez",
        "Moore",
        "Martin",
        "Jackson",
        "Thompson",
        "White",
        "Lopez",
        "Lee",
        "Gonzalez",
        "Harris",
        "Clark",
        "Lewis",
        "Robinson",
        "Walker",
        "Perez",
        "Hall",
        "Young",
        "Allen",
        "Sanchez",
        "Wright",
        "King",
        "Scott",
        "Green",
        "Baker",
        "Adams",
        "Nelson",
        "Hill",
        "Ramirez",
        "Campbell",
        "Mitchell",
        "Roberts",
        "Carter",
        "Phillips",
        "Evans",
        "Turner",
        "Torres",
        "Parker",
        "Collins",
        "Edwards",
        "Stewart",
        "Flores",
        "Morris",
        "Nguyen",
        "Murphy",
        "Rivera",
        "Cook",
        "Rogers",
        "Morgan",
        "Peterson",
        "Cooper",
        "Reed",
        "Bailey",
        "Bell",
        "Gomez",
        "Kelly",
        "Howard",
        "Ward",
        "Cox",
        "Diaz",
        "Richardson",
        "Wood",
        "Watson",
        "Brooks",
        "Bennett",
        "Gray",
        "James",
        "Reyes",
        "Cruz",
        "Hughes",
        "Price",
        "Myers",
        "Long",
        "Foster",
        "Sanders",
        "Ross",
        "Morales",
        "Powell",
        "Sullivan",
        "Russell",
        "Ortiz",
        "Jenkins",
        "Gutierrez",
        "Perry",
        "Butler",
        "Barnes",
        "Fisher",
    ]

    # Institutions
    INSTITUTIONS = [
        "MIT",
        "Stanford University",
        "Carnegie Mellon University",
        "UC Berkeley",
        "Oxford University",
        "Cambridge University",
        "ETH Zurich",
        "EPFL",
        "University of Toronto",
        "McGill University",
        "Tsinghua University",
        "National University of Singapore",
        "Tokyo Institute of Technology",
        "Indian Institute of Technology",
        "KAIST",
        "Technion",
        "TU Munich",
        "Imperial College London",
        "UCL",
        "Harvard University",
        "Yale University",
        "Princeton University",
        "Columbia University",
        "NYU",
        "UCLA",
        "UCSD",
        "University of Washington",
        "Georgia Tech",
        "UT Austin",
        "UIUC",
        "University of Michigan",
        "Cornell University",
        "University of Pennsylvania",
        "Johns Hopkins University",
        "Duke University",
        "Northwestern University",
    ]

    # Research topics within LLM wargaming domain
    RESEARCH_TOPICS = [
        {
            "area": "strategic_planning",
            "templates": [
                "Strategic Planning with {llm_type} in {game_context}",
                "Multi-Agent {llm_type} Systems for Strategic {scenario}",
                "{llm_type}-Assisted Decision Making in {game_context}",
                "Evaluating {llm_type} Performance in Strategic {scenario}",
            ],
        },
        {
            "area": "human_ai_teaming",
            "templates": [
                "Human-{llm_type} Collaboration in {game_context}",
                "Augmenting Human Decision-Making with {llm_type} in {scenario}",
                "Trust and Transparency in Human-{llm_type} {game_context}",
                "Cognitive Load Reduction through {llm_type} Support in {scenario}",
            ],
        },
        {
            "area": "adversarial_reasoning",
            "templates": [
                "Adversarial Reasoning with {llm_type} Agents",
                "Red Team Simulation using {llm_type} for {scenario}",
                "Competitive Multi-Agent {llm_type} in {game_context}",
                "Deception Detection and Generation in {llm_type} {scenario}",
            ],
        },
        {
            "area": "scenario_generation",
            "templates": [
                "Automated Scenario Generation using {llm_type}",
                "Dynamic {scenario} Creation with {llm_type} Models",
                "Procedural Wargame Design through {llm_type} Generation",
                "Narrative Coherence in {llm_type}-Generated {game_context}",
            ],
        },
        {
            "area": "analysis_insights",
            "templates": [
                "Post-Game Analysis using {llm_type} for {scenario}",
                "Pattern Recognition in {game_context} with {llm_type}",
                "Automated After-Action Review Generation via {llm_type}",
                "Strategic Insight Extraction from {scenario} using {llm_type}",
            ],
        },
    ]

    LLM_TYPES = [
        "GPT-4",
        "Claude-3",
        "PaLM-2",
        "LLaMA-2",
        "Gemini",
        "Mistral",
        "Falcon",
        "BLOOM",
        "ChatGPT",
        "Bard",
        "Large Language Models",
        "Foundation Models",
        "Transformer Models",
        "Neural Language Models",
    ]

    GAME_CONTEXTS = [
        "Military Wargames",
        "Crisis Management Simulations",
        "Business Strategy Games",
        "Diplomatic Negotiations",
        "Cybersecurity Exercises",
        "Emergency Response Drills",
        "Political Simulations",
        "Economic War Games",
        "Supply Chain Disruption Scenarios",
        "Pandemic Response Games",
    ]

    SCENARIOS = [
        "Conflict Escalation",
        "Resource Allocation",
        "Coalition Formation",
        "Information Warfare",
        "Humanitarian Crisis",
        "Trade Negotiations",
        "Border Disputes",
        "Terrorism Response",
        "Natural Disaster Management",
        "Technology Competition",
        "Space Warfare",
        "Arctic Sovereignty",
    ]

    # Abstract components
    ABSTRACT_INTROS = [
        "We present a novel framework for integrating {llm_type} into {game_context}, addressing the challenge of {challenge}.",
        "This paper investigates the application of {llm_type} in {scenario}, focusing on {focus_area}.",
        "Recent advances in {llm_type} have opened new possibilities for {application}. We explore {specific_aspect}.",
        "The intersection of artificial intelligence and {domain} presents unique opportunities. This work examines {research_question}.",
        "As {llm_type} capabilities continue to evolve, their potential in {game_context} remains underexplored. We propose {approach}.",
    ]

    CHALLENGES = [
        "maintaining strategic coherence across multiple turns",
        "balancing computational efficiency with decision quality",
        "ensuring robust performance under adversarial conditions",
        "preserving human agency while leveraging AI assistance",
        "managing uncertainty and incomplete information",
        "scaling to complex multi-stakeholder scenarios",
        "preventing exploitation of AI decision patterns",
        "maintaining ethical constraints in autonomous systems",
    ]

    FOCUS_AREAS = [
        "real-time decision support mechanisms",
        "human-AI trust dynamics",
        "explainable AI for strategic planning",
        "multi-modal information integration",
        "adaptive learning from game outcomes",
        "cross-domain transfer of strategic knowledge",
        "robustness to adversarial manipulation",
        "cognitive load optimization",
    ]

    METHODS = [
        "reinforcement learning with human feedback",
        "chain-of-thought prompting strategies",
        "multi-agent debate frameworks",
        "hierarchical planning architectures",
        "Bayesian uncertainty quantification",
        "graph neural networks for relationship modeling",
        "transformer-based sequence modeling",
        "federated learning approaches",
    ]

    RESULTS = [
        "outperformed baseline methods by {improvement}% in {metric}",
        "achieved {accuracy}% accuracy in {task} while maintaining {property}",
        "demonstrated significant improvements in {outcome} compared to {baseline}",
        "showed robust performance across {num} different {scenario_type}",
        "reduced {metric} by {reduction}% while improving {other_metric}",
    ]

    # Game types
    GAME_TYPES = [
        "Matrix Game",
        "Tabletop Exercise",
        "Digital Simulation",
        "Seminar Game",
        "Committee Game",
        "Free Kriegsspiel",
        "Rigid Kriegsspiel",
        "Computer-Assisted Game",
        "Hybrid Simulation",
        "Educational Wargame",
        "Training Simulation",
        "Research Game",
    ]

    # Failure modes
    FAILURE_MODES = [
        "Hallucination",
        "Context Window Limitations",
        "Reasoning Errors",
        "Bias Amplification",
        "Adversarial Manipulation",
        "Coherence Breakdown",
        "Strategic Myopia",
        "Overconfidence",
        "Underspecification",
        "Temporal Inconsistency",
        "Coalition Instability",
        "Goal Misalignment",
    ]

    # Venues
    CONFERENCES = [
        ("NeurIPS", "Conference on Neural Information Processing Systems"),
        ("ICML", "International Conference on Machine Learning"),
        ("ICLR", "International Conference on Learning Representations"),
        ("AAAI", "AAAI Conference on Artificial Intelligence"),
        ("IJCAI", "International Joint Conference on Artificial Intelligence"),
        ("ACL", "Annual Meeting of the Association for Computational Linguistics"),
        ("EMNLP", "Conference on Empirical Methods in Natural Language Processing"),
        (
            "AAMAS",
            "International Conference on Autonomous Agents and Multiagent Systems",
        ),
        ("CoG", "IEEE Conference on Games"),
        ("ICIDS", "International Conference on Interactive Digital Storytelling"),
    ]

    JOURNALS = [
        "Nature Machine Intelligence",
        "Journal of Artificial Intelligence Research",
        "Artificial Intelligence",
        "IEEE Transactions on Games",
        "Simulation & Gaming",
        "Journal of Defense Modeling and Simulation",
        "Computers in Human Behavior",
        "Decision Support Systems",
        "International Journal of Human-Computer Studies",
    ]

    def __init__(self, seed: int | None = None):
        """Initialize generator with optional seed for reproducibility."""
        if seed is not None:
            random.seed(seed)

    def generate_author_name(self) -> str:
        """Generate a realistic author name."""
        first = random.choice(self.FIRST_NAMES)
        last = random.choice(self.LAST_NAMES)

        # Sometimes add middle initial
        if random.random() < 0.3:
            middle = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + "."
            return f"{first} {middle} {last}"
        return f"{first} {last}"

    def generate_authors(self, min_authors: int = 1, max_authors: int = 6) -> list[str]:
        """Generate a list of author names."""
        num_authors = random.randint(min_authors, max_authors)
        authors = []

        for _ in range(num_authors):
            author = self.generate_author_name()
            # Sometimes add institution affiliation
            if random.random() < 0.7:
                institution = random.choice(self.INSTITUTIONS)
                author = f"{author} ({institution})"
            authors.append(author)

        return authors

    def generate_title(self) -> str:
        """Generate a realistic paper title."""
        topic = random.choice(self.RESEARCH_TOPICS)
        template = random.choice(topic["templates"])

        title = template.format(
            llm_type=random.choice(self.LLM_TYPES),
            game_context=random.choice(self.GAME_CONTEXTS),
            scenario=random.choice(self.SCENARIOS),
        )

        # Sometimes add a subtitle
        if random.random() < 0.3:
            subtitles = [
                "A Comparative Study",
                "An Empirical Analysis",
                "A Case Study",
                "Lessons from Practice",
                "A Multi-Agent Perspective",
                "Challenges and Opportunities",
                "A Framework for Analysis",
                "Experimental Results",
            ]
            title += f": {random.choice(subtitles)}"

        return title

    def generate_abstract(self, title: str) -> str:
        """Generate a realistic abstract based on the title."""
        # Extract context from title
        llm_type = "large language models"
        game_context = "strategic simulations"

        for llm in self.LLM_TYPES:
            if llm.lower() in title.lower():
                llm_type = llm
                break

        for context in self.GAME_CONTEXTS:
            if context.lower() in title.lower():
                game_context = context
                break

        # Build abstract
        intro = random.choice(self.ABSTRACT_INTROS).format(
            llm_type=llm_type,
            game_context=game_context,
            challenge=random.choice(self.CHALLENGES),
            focus_area=random.choice(self.FOCUS_AREAS),
            application=f"{llm_type} in {game_context}",
            specific_aspect=random.choice(self.FOCUS_AREAS),
            domain=game_context.split()[0],
            research_question=f"how {llm_type} can enhance {game_context}",
            approach=f"a {random.choice(self.METHODS)}-based approach",
            scenario=random.choice(self.SCENARIOS),
        )

        # Add methodology
        method = (
            f"Our approach employs {random.choice(self.METHODS)} combined with "
            f"{random.choice(['domain-specific fine-tuning', 'prompt engineering',
                                  'multi-modal integration', 'hierarchical planning'])}. "
        )

        # Add experiment description
        experiment = (
            f"We evaluate our system through {random.randint(3, 12)} "
            f"{random.choice(['controlled experiments', 'case studies',
                                     'simulation runs', 'user studies'])} involving "
            f"{random.randint(10, 200)} {random.choice(['participants',
                                                                'scenarios', 'test cases'])}. "
        )

        # Add results
        result_template = random.choice(self.RESULTS)
        results = result_template.format(
            improvement=random.randint(10, 40),
            accuracy=random.randint(75, 95),
            metric=random.choice(
                ["decision quality", "response time", "strategic coherence"]
            ),
            task=random.choice(
                ["strategic planning", "resource allocation", "threat assessment"]
            ),
            property=random.choice(["interpretability", "robustness", "efficiency"]),
            outcome=random.choice(
                ["decision accuracy", "planning efficiency", "team performance"]
            ),
            baseline=random.choice(
                ["human-only teams", "rule-based systems", "previous SOTA"]
            ),
            num=random.randint(5, 20),
            scenario_type=random.choice(
                ["scenarios", "game configurations", "difficulty levels"]
            ),
            reduction=random.randint(20, 60),
            other_metric=random.choice(
                ["accuracy", "user satisfaction", "completion time"]
            ),
        )
        results = "Our results " + results + ". "

        # Add conclusion
        conclusions = [
            f"These findings suggest that {llm_type} can significantly enhance {game_context} "
            f"when properly integrated with {random.choice(['human expertise', 'domain knowledge',
                                                          'existing frameworks'])}.",
            f"This work demonstrates the potential of {llm_type} to transform {game_context} "
            f"while highlighting important {random.choice(['limitations', 'considerations',
                                                          'challenges'])} for future research.",
            f"Our contributions provide a foundation for further exploration of {llm_type} "
            f"in {random.choice(['complex decision-making', 'strategic planning',
                                'adversarial scenarios'])}.",
        ]
        conclusion = random.choice(conclusions)

        return intro + " " + method + experiment + results + conclusion

    def generate_doi(self, year: int) -> str:
        """Generate a realistic DOI."""
        publishers = ["10.1109", "10.1145", "10.1016", "10.1007", "10.1038", "10.48550"]
        publisher = random.choice(publishers)

        # Generate realistic suffix
        if publisher == "10.48550":  # arXiv
            return f"{publisher}/arXiv.{year}.{random.randint(1000, 99999):05d}"
        else:
            journal_code = random.choice(["ABC", "DEF", "GHI", "JKL", "MNO", "PQR"])
            return f"{publisher}/{journal_code}.{year}.{random.randint(100000, 999999)}"

    def generate_arxiv_id(self, year: int) -> str:
        """Generate a realistic arXiv ID."""
        # Extract last two digits of year
        year_suffix = str(year)[-2:]

        # Generate month (01-12)
        month = random.randint(1, 12)

        # Generate paper number
        paper_num = random.randint(1, 30000)

        # Choose category
        categories = ["cs.AI", "cs.CL", "cs.MA", "cs.GT", "cs.HC", "cs.CY"]

        return f"{year_suffix}{month:02d}.{paper_num:05d}"

    def generate_publication_date(self, year: int) -> datetime:
        """Generate a random date within the given year."""
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)

        time_between = end_date - start_date
        days_between = time_between.days

        random_days = random.randint(0, days_between)
        return start_date + timedelta(days=random_days)

    def generate_venue(self) -> dict[str, str]:
        """Generate venue information."""
        if random.random() < 0.6:  # 60% conferences
            conf = random.choice(self.CONFERENCES)
            return {"venue": conf[0], "venue_full": conf[1], "venue_type": "conference"}
        else:  # 40% journals
            journal = random.choice(self.JOURNALS)
            return {"venue": journal, "venue_full": journal, "venue_type": "journal"}

    def generate_paper(self, year: int | None = None) -> dict[str, Any]:
        """Generate a complete realistic paper entry."""
        if year is None:
            year = random.randint(2020, 2024)

        title = self.generate_title()
        authors = self.generate_authors()
        abstract = self.generate_abstract(title)
        pub_date = self.generate_publication_date(year)
        venue_info = self.generate_venue()

        # Generate identifiers - not all papers have all IDs
        paper = {
            "title": title,
            "authors": authors,
            "author_list": "; ".join(authors),
            "abstract": abstract,
            "year": year,
            "published": pub_date.isoformat(),
            "source_db": random.choice(
                ["arxiv", "semantic_scholar", "crossref", "pubmed"]
            ),
            **venue_info,
        }

        # Add identifiers based on source
        if paper["source_db"] == "arxiv" or random.random() < 0.7:
            paper["arxiv_id"] = self.generate_arxiv_id(year)
            paper["pdf_url"] = f"https://arxiv.org/pdf/{paper['arxiv_id']}.pdf"

        if paper["source_db"] != "arxiv" or random.random() < 0.5:
            paper["doi"] = self.generate_doi(year)

        # Add categories
        paper["categories"] = random.sample(
            ["cs.AI", "cs.CL", "cs.MA", "cs.GT", "cs.HC", "cs.CY", "cs.LG"],
            k=random.randint(1, 3),
        )

        # Add game-specific metadata
        paper["game_type"] = random.choice(self.GAME_TYPES)
        paper["llm_family"] = random.choice(
            ["GPT", "Claude", "LLaMA", "PaLM", "Gemini", "Mistral", "Other"]
        )

        # Add failure modes (not all papers discuss failures)
        if random.random() < 0.7:
            num_failures = random.randint(1, 4)
            paper["failure_modes"] = "; ".join(
                random.sample(self.FAILURE_MODES, k=num_failures)
            )
        else:
            paper["failure_modes"] = None

        # Add analysis metadata
        paper["awscale"] = random.randint(1, 5)  # Analytical-Wild/Creative scale
        paper["open_ended"] = random.choice(["yes", "no"])
        paper["quantitative"] = random.choice(["yes", "no"])

        # Add quality indicators
        paper["citations"] = int(
            random.lognormvariate(2, 1.5)
        )  # Log-normal distribution
        paper["influential_citations"] = max(
            0, int(paper["citations"] * random.uniform(0.05, 0.2))
        )

        return paper

    def generate_paper_batch(
        self, count: int, year_range: tuple | None = None
    ) -> list[dict[str, Any]]:
        """Generate multiple papers with consistent distribution."""
        papers = []

        if year_range:
            years = list(range(year_range[0], year_range[1] + 1))
        else:
            years = [2020, 2021, 2022, 2023, 2024]

        for i in range(count):
            # Distribute papers across years with recent bias
            weights = [1, 1.5, 2, 2.5, 3]  # More recent papers
            if len(years) == len(weights):
                year = random.choices(years, weights=weights)[0]
            else:
                year = random.choice(years)

            paper = self.generate_paper(year)
            paper["batch_id"] = i
            papers.append(paper)

        return papers

    def generate_extraction_results(self, paper: dict[str, Any]) -> dict[str, Any]:
        """Generate realistic extraction results for a paper."""
        extraction = {
            "research_questions": self._generate_research_questions(paper),
            "key_contributions": self._generate_contributions(paper),
            "simulation_approach": self._generate_simulation_approach(paper),
            "llm_usage": self._generate_llm_usage(paper),
            "human_llm_comparison": self._generate_comparison(),
            "evaluation_metrics": self._generate_metrics(),
            "prompting_strategies": self._generate_prompting_strategies(),
            "emerging_behaviors": self._generate_emerging_behaviors(),
            "datasets_used": self._generate_datasets(),
            "limitations": self._generate_limitations(),
            "future_work": self._generate_future_work(),
            "ethical_considerations": self._generate_ethical_considerations(),
        }

        # Add metadata
        extraction["extraction_timestamp"] = datetime.now().isoformat()
        extraction["extraction_model"] = random.choice(
            ["gemini/gemini-pro", "gpt-3.5-turbo"]
        )
        extraction["extraction_confidence"] = random.uniform(0.7, 0.95)

        return extraction

    def _generate_research_questions(self, paper: dict[str, Any]) -> str:
        """Generate research questions based on paper content."""
        questions = [
            f"How can {paper.get('llm_family', 'LLM')} models enhance {paper.get('game_type', 'wargaming')}?",
            f"What are the limitations of current AI systems in {random.choice(self.SCENARIOS)}?",
            "How do human-AI teams perform compared to human-only teams in strategic decision-making?",
            "Can LLMs maintain coherent long-term strategies in multi-turn games?",
            "What prompting strategies optimize LLM performance in adversarial scenarios?",
        ]

        # Pick 1-3 questions
        selected = random.sample(questions, k=random.randint(1, 3))
        return " ".join(f"RQ{i+1}: {q}" for i, q in enumerate(selected))

    def _generate_contributions(self, paper: dict[str, Any]) -> str:
        """Generate key contributions."""
        contributions = [
            f"Novel {random.choice(self.METHODS)} framework for {paper.get('game_type', 'wargaming')}",
            f"Comprehensive evaluation across {random.randint(5, 20)} different scenarios",
            "Open-source implementation and benchmark dataset",
            "Theoretical analysis of LLM decision-making in strategic contexts",
            f"Human subject study with {random.randint(20, 100)} participants",
            f"New metrics for evaluating AI performance in {random.choice(self.GAME_CONTEXTS)}",
        ]

        # Pick 2-4 contributions
        selected = random.sample(contributions, k=random.randint(2, 4))
        return "; ".join(selected)

    def _generate_simulation_approach(self, paper: dict[str, Any]) -> str:
        """Generate simulation approach description."""
        approaches = [
            f"{paper.get('game_type', 'Digital simulation')} with {random.choice(['turn-based', 'real-time', 'asynchronous'])} gameplay",
            f"Multi-agent system with {random.randint(2, 10)} autonomous LLM agents",
            f"Human-in-the-loop simulation with AI assistance at {random.choice(['planning', 'execution', 'analysis'])} phase",
            f"Hierarchical decision-making with {random.choice(['centralized', 'decentralized', 'hybrid'])} control",
            f"Scenario-based testing across {random.choice(['military', 'economic', 'diplomatic', 'hybrid'])} domains",
        ]

        return random.choice(approaches)

    def _generate_llm_usage(self, paper: dict[str, Any]) -> str:
        """Generate LLM usage description."""
        usages = [
            "Autonomous agents making independent decisions",
            "Decision support system providing recommendations to human players",
            "Scenario generation and dynamic adaptation",
            "Natural language interface for complex commands",
            "Multi-modal reasoning combining text, maps, and data",
            "Adversarial red team simulation",
            "After-action analysis and insight generation",
        ]

        # Pick 1-2 uses
        selected = random.sample(usages, k=random.randint(1, 2))
        return "; ".join(selected)

    def _generate_comparison(self) -> str:
        """Generate human-LLM comparison results."""
        comparisons = [
            "Humans showed better long-term strategic thinking while LLMs excelled at rapid tactical analysis",
            "LLMs demonstrated more consistent performance but lacked creative problem-solving of human experts",
            "Human-AI teams outperformed both human-only and AI-only configurations by 25%",
            "LLMs showed less cognitive bias but were more susceptible to adversarial manipulation",
            "Performance parity achieved in structured scenarios; humans superior in novel situations",
        ]

        return random.choice(comparisons)

    def _generate_metrics(self) -> list[str]:
        """Generate evaluation metrics."""
        all_metrics = [
            "Decision quality score",
            "Time to decision",
            "Strategic coherence",
            "Resource efficiency",
            "Goal achievement rate",
            "Adaptability index",
            "Explanation quality",
            "Human trust rating",
            "Cognitive load reduction",
            "Error rate",
            "Learning curve",
            "Robustness score",
            "Fairness metrics",
        ]

        return random.sample(all_metrics, k=random.randint(3, 6))

    def _generate_prompting_strategies(self) -> str:
        """Generate prompting strategies description."""
        strategies = [
            "Chain-of-thought prompting with explicit reasoning steps",
            "Few-shot examples from expert human players",
            "Role-based prompting with detailed persona descriptions",
            "Hierarchical prompting for complex multi-step decisions",
            "Adversarial prompting to test robustness",
            "Self-consistency voting across multiple generations",
            "Dynamic prompt adaptation based on game state",
        ]

        # Pick 1-3 strategies
        selected = random.sample(strategies, k=random.randint(1, 3))
        return "; ".join(selected)

    def _generate_emerging_behaviors(self) -> str:
        """Generate emerging behaviors description."""
        behaviors = [
            "Spontaneous coalition formation between AI agents",
            "Development of novel strategies not seen in training",
            "Deceptive communication patterns in competitive scenarios",
            "Risk-averse behavior in high-stakes situations",
            "Convergence to Nash equilibrium in repeated games",
            "Creative exploitation of game rule ambiguities",
            "Emergent specialization in multi-agent teams",
        ]

        return random.choice(behaviors)

    def _generate_datasets(self) -> list[str]:
        """Generate datasets used."""
        datasets = [
            "Custom wargame scenario collection (n=1000)",
            "Historical military conflict database",
            "RAND wargaming reports corpus",
            "Synthetic game traces from rule-based agents",
            "Human expert gameplay recordings",
            "Multi-modal military doctrine documents",
            "Open-source strategy game replays",
        ]

        return random.sample(datasets, k=random.randint(1, 3))

    def _generate_limitations(self) -> str:
        """Generate limitations description."""
        limitations = [
            "Limited to text-based interaction without visual/spatial reasoning",
            "Computational constraints prevent real-time application at scale",
            "Evaluation limited to specific game types and scenarios",
            "Potential for adversarial exploitation not fully explored",
            "Generalization to out-of-distribution scenarios remains uncertain",
            "Ethical implications of autonomous strategic decisions need further study",
        ]

        # Pick 1-2 limitations
        selected = random.sample(limitations, k=random.randint(1, 2))
        return "; ".join(selected)

    def _generate_future_work(self) -> str:
        """Generate future work description."""
        future = [
            "Extend to multi-modal reasoning with maps and imagery",
            "Investigate transfer learning across different game domains",
            "Develop robust defenses against adversarial manipulation",
            "Scale to massively multi-agent scenarios",
            "Integrate with physical simulation engines",
            "Explore human-AI collaboration protocols",
            "Address ethical and safety considerations",
        ]

        # Pick 2-3 items
        selected = random.sample(future, k=random.randint(2, 3))
        return "; ".join(selected)

    def _generate_ethical_considerations(self) -> str:
        """Generate ethical considerations."""
        considerations = [
            "Dual-use nature of strategic AI systems",
            "Transparency and explainability requirements",
            "Human oversight and meaningful control",
            "Bias mitigation in conflict scenarios",
            "International governance frameworks needed",
            "Potential for automation bias in critical decisions",
        ]

        return random.choice(considerations)
