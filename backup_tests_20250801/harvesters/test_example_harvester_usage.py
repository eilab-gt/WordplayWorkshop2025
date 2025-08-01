"""Example test file demonstrating usage of test utilities and fixtures."""

from unittest.mock import patch

from tests.test_fixtures import (
    create_arxiv_response,
    create_crossref_response,
    create_semantic_scholar_response,
)
from tests.test_utils import (
    assert_papers_filtered_by_year,
    assert_rate_limiting_applied,
    create_mock_response,
    generate_test_papers,
    mock_requests_get,
    mock_time_sleep,
)


class TestExampleHarvesterUsage:
    """Example test class showing how to use test utilities."""

    def test_using_mock_config(self, mock_config):
        """Example of using the mock_config fixture."""
        # The mock_config fixture provides a fully configured MockConfig object
        assert mock_config.search_years == (2018, 2025)
        assert mock_config.unpaywall_email == "test@example.com"
        assert "crossref" in mock_config.rate_limits
        assert mock_config.rate_limits["crossref"]["delay_milliseconds"] == 10

    def test_using_sample_papers(self, sample_paper, sample_papers):
        """Example of using sample paper fixtures."""
        # Single paper
        assert sample_paper.title == "LLM-powered Wargaming Simulation"
        assert sample_paper.year == 2024
        assert len(sample_paper.authors) == 2

        # Multiple papers
        assert len(sample_papers) == 3
        assert sample_papers[0].source_db == "test"

        # Papers can be filtered
        recent_papers = [p for p in sample_papers if p.year >= 2023]
        assert len(recent_papers) == 2

    def test_using_mock_responses(self):
        """Example of creating mock API responses."""
        # Create an arXiv response
        arxiv_response = create_arxiv_response(
            [
                {
                    "arxiv_id": "2401.12345",
                    "title": "Test Paper",
                    "authors": ["John Doe", "Jane Smith"],
                    "abstract": "Test abstract",
                    "year": 2024,
                }
            ]
        )
        assert len(arxiv_response["entries"]) == 1
        assert arxiv_response["entries"][0]["title"] == "Test Paper"

        # Create a CrossRef response
        crossref_response = create_crossref_response(
            [
                {
                    "doi": "10.1234/test",
                    "title": "CrossRef Test Paper",
                    "authors": ["Alice Johnson"],
                    "year": 2023,
                    "venue": "Test Journal",
                }
            ]
        )
        assert crossref_response["status"] == "ok"
        assert len(crossref_response["message"]["items"]) == 1

        # Create a Semantic Scholar response
        ss_response = create_semantic_scholar_response(
            [
                {
                    "paper_id": "abc123",
                    "title": "Semantic Scholar Paper",
                    "authors": ["Bob Wilson"],
                    "year": 2024,
                    "citations": 10,
                }
            ]
        )
        assert ss_response["total"] == 1
        assert ss_response["data"][0]["citationCount"] == 10

    def test_using_context_managers(self):
        """Example of using context manager utilities."""
        # Mock time.sleep for rate limiting tests
        with mock_time_sleep() as mock_sleep:
            # Simulate code that calls time.sleep
            import time

            time.sleep(0.01)
            time.sleep(0.01)

            # Assert rate limiting was applied
            assert mock_sleep.call_count == 2
            assert_rate_limiting_applied(mock_sleep, expected_calls=2)

        # Mock requests.get with predefined responses
        mock_response = create_mock_response(
            status_code=200, json_data={"result": "success"}
        )

        with mock_requests_get([mock_response]) as mock_get:
            import requests

            response = requests.get("https://api.example.com/test")

            assert response.status_code == 200
            assert response.json()["result"] == "success"
            assert mock_get.call_count == 1

    def test_year_filtering_utilities(self):
        """Example of using year filtering utilities."""
        papers = generate_test_papers(10, year_range=(2020, 2025))

        # Filter to recent papers
        filtered = [p for p in papers if p.year >= 2023]

        # Assert all papers are within range
        assert_papers_filtered_by_year(filtered, 2023, 2025)

    def test_using_llm_service_mock(self, mock_llm_service):
        """Example of using the mock LLM service."""
        # Check health
        with patch("requests.get", side_effect=mock_llm_service.mock_health_check):
            import requests

            response = requests.get("http://localhost:8000/health")
            assert response.status_code == 200

        # Get models
        with patch("requests.get", side_effect=mock_llm_service.mock_get_models):
            response = requests.get("http://localhost:8000/models")
            models = response.json()
            assert "gemini/gemini-pro" in models
            assert models["gemini/gemini-pro"]["available"]

        # Extract data
        with patch("requests.post", side_effect=mock_llm_service.mock_extract):
            response = requests.post("http://localhost:8000/extract", json={})
            result = response.json()
            assert result["success"]
            assert "research_questions" in result["extracted_data"]

    def test_paper_generation(self):
        """Example of generating test papers."""
        # Generate papers for different sources
        arxiv_papers = generate_test_papers(5, source_db="arxiv")
        crossref_papers = generate_test_papers(5, source_db="crossref")

        assert all(p.source_db == "arxiv" for p in arxiv_papers)
        assert all(p.source_db == "crossref" for p in crossref_papers)

        # Papers have varied data
        assert len(set(p.year for p in arxiv_papers)) > 1
        assert len(set(p.venue for p in arxiv_papers)) > 1

        # Some papers have DOIs, some have arXiv IDs
        assert any(p.doi for p in arxiv_papers)
        assert any(p.arxiv_id for p in arxiv_papers)

    def test_mock_api_errors(self):
        """Example of testing error scenarios."""
        # Test 404 response
        error_response = create_mock_response(status_code=404, raise_for_status=True)

        with mock_requests_get([error_response]) as mock_get:
            import requests

            # This would normally raise an exception
            try:
                response = requests.get("https://api.example.com/notfound")
                response.raise_for_status()
                assert False, "Should have raised exception"
            except Exception as e:
                assert "HTTP 404" in str(e)

    def test_combined_workflow(self, mock_config):
        """Example of a complete test workflow using multiple utilities."""
        # 1. Generate test data
        test_papers = generate_test_papers(20, year_range=(2020, 2024))

        # 2. Create mock API responses
        api_responses = []
        for batch in [test_papers[:10], test_papers[10:]]:
            response_data = create_crossref_response(
                [
                    {
                        "doi": p.doi,
                        "title": p.title,
                        "authors": p.authors,
                        "year": p.year,
                        "venue": p.venue,
                    }
                    for p in batch
                    if p.doi
                ]
            )
            api_responses.append(create_mock_response(200, response_data))

        # 3. Test with mocked requests and rate limiting
        with mock_requests_get(api_responses) as mock_get:
            with mock_time_sleep() as mock_sleep:
                # Simulate harvester behavior
                all_results = []
                for response in api_responses:
                    # Simulate API call
                    import time

                    import requests

                    resp = requests.get("https://api.crossref.org/works")
                    data = resp.json()
                    all_results.extend(data["message"]["items"])

                    # Simulate rate limiting
                    time.sleep(0.01)

                # Verify results
                assert len(all_results) > 0
                assert mock_get.call_count == 2
                assert_rate_limiting_applied(mock_sleep, expected_calls=2)
