#include "DuckDuckGoEngine.hpp"

#include <curl/curl.h>
#include <regex>
#include <iostream>

static size_t writeCallback(void* contents, size_t size, size_t nmemb, void* userp)
{
    size_t total = size * nmemb;
    std::string* str = static_cast<std::string*>(userp);
    str->append(static_cast<char*>(contents), total);
    return total;
}

std::string DuckDuckGoEngine::fetch(const std::string& url)
{
    CURL* curl = curl_easy_init();
    std::string response;

    if (!curl)
        return response;

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    // MUITO IMPORTANTE (sem isso DDG quebra)
    curl_easy_setopt(
        curl,
        CURLOPT_USERAGENT,
        "Mozilla/5.0 (X11; Linux x86_64) BareNet/0.1"
    );

    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 15L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        std::cerr << "[DuckDuckGo] curl error: "
                  << curl_easy_strerror(res) << "\n";
    }

    curl_easy_cleanup(curl);
    return response;
}

std::vector<SearchResult> DuckDuckGoEngine::parseResults(
    const std::string& html,
    int limit
) {
    std::vector<SearchResult> results;

    // DuckDuckGo Lite links
    std::regex linkRegex(
        R"(<a rel="nofollow" class="result-link" href="([^"]+)")"
    );

    auto begin = std::sregex_iterator(html.begin(), html.end(), linkRegex);
    auto end = std::sregex_iterator();

    for (auto it = begin; it != end && (int)results.size() < limit; ++it) {
        SearchResult r;
        r.url = (*it)[1];
        r.title = r.url;
        r.body = "";
        r.status_code = 0;

        results.push_back(r);
    }

    return results;
}

std::vector<SearchResult> DuckDuckGoEngine::search(
    const std::string& query,
    int limit
) {
    std::string url =
        "https://lite.duckduckgo.com/lite/?q=" + query;

    std::string html = fetch(url);

    // DEBUG TEMPORÁRIO (recomendo manter por enquanto)
    std::cerr << "[DuckDuckGo] HTML size: " << html.size() << "\n";

    if (html.empty())
        return {};

    return parseResults(html, limit);
}

std::string DuckDuckGoEngine::name() const
{
    return "DuckDuckGo";
}
