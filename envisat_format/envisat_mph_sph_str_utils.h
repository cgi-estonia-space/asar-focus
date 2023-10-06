#pragma once

#include <cstdlib>
#include <string>

#include <boost/date_time/posix_time/posix_time.hpp>

#include "envisat_lvl1_ads.h"
#include "util/checks.h"

// Utility functions to generate strings for MPH and SPH

// Names and sizes from
// ENVISAT-1 PRODUCTS SPECIFICATIONS
// ANNEX A: PRODUCT DATA CONVENTIONS
// IDEAS+-SER-IPF-SPE-2337

template <size_t N>
void LastNewline(uc (&arr)[N]) {
    arr[N - 1] = '\n';
}

template <size_t N>
void ClearSpare(uc (&arr)[N]) {
    memset(arr, ' ', N - 1);
    arr[N - 1] = '\n';
}

template <size_t N>
void CopyStr(uc (&arr)[N], const std::string& s) {
    if (N != s.size()) {
        printf("sizes dest src = %zu %zu\n", N, s.size());
        printf("BAD STRING!? = %s\n", s.c_str());
        CHECK_BOOL(false);
    }
    memcpy(&arr[0], s.data(), N);
}

template <size_t N>
void SetStr(uc (&arr)[N], const char* keyword, std::string value) {
    std::string buf = keyword;
    buf += "=\"";
    size_t value_sz = N - buf.size() - 2;
    value.resize(value_sz, ' ');
    buf += value;
    buf += "\"\n";
    CopyStr(arr, buf);
}

template <size_t N>
void SetChar(uc (&arr)[N], const char* keyword, char value) {
    std::string buf = keyword;
    buf += "=";
    buf += value;
    buf += '\n';
    CopyStr(arr, buf);
}

inline std::string ToAs(int val) {
    char buf[20];
    snprintf(buf, 20, "%+06d", val); // Sign specifier counts as additional integer part count.
    return buf;
}

inline std::string ToAl(int val) {
    char buf[20];
    snprintf(buf, 20, "%+011d", val); // Sign specifier counts as additional integer part count.
    return buf;
}

inline std::string ToAc(int val) {
    char buf[20];
    snprintf(buf, 20, "%+04d", val); // Sign specifier counts as additional integer part count.
    return buf;
}

inline std::string ToAd(int64_t val) {
    char buf[30];
    snprintf(buf, 30, "%+021ld", val); // Sign specifier counts as additional integer part count.
    return buf;
}

inline std::string ToAdo06(double val) {
    // TODO - Something should check if integer part is 0 actually.
    char buf[10]; // printf() somehow always inserts the integer part as well.
    snprintf(buf, 10, "%+.6f", val);
    memmove(buf + 1, buf + 2, 8); // Replace integer part
    return buf;
}

inline std::string ToAdo73(double val) {
    char buf[20];
    snprintf(buf, 20, "%+012.3f", val); // Sign specifier counts as additional integer part count.
    return buf;
}

inline std::string ToAdo46(double val) {
    char buf[20];
    snprintf(buf, 20, "%+012.6f", val); // Sign specifier counts as additional integer part count.
    return buf;
}

inline std::string ToAfl(float val) {
    char buf[20];
    float exp = log10f(val);
    (void)exp;
    snprintf(buf, 20, "%+.8E", val);
    return buf;
}

template <size_t N>
void SetAs(uc (&arr)[N], const char* keyword, int val, const char* unit = "") {
    std::string buf = keyword;
    buf += "=";
    buf += ToAs(val);
    buf += unit;
    buf += '\n';
    CopyStr(arr, buf);
}

template <size_t N>
void SetAl(uc (&arr)[N], const char* keyword, int val, const char* unit = "") {
    std::string buf = keyword;
    buf += "=";
    buf += ToAl(val);
    buf += unit;
    buf += '\n';
    CopyStr(arr, buf);
}

template <size_t N>
void SetAc(uc (&arr)[N], const char* keyword, int val, const char* unit = "") {
    std::string buf = keyword;
    buf += "=";
    buf += ToAc(val);
    buf += unit;
    buf += '\n';
    CopyStr(arr, buf);
}

template <size_t N>
void SetAd(uc (&arr)[N], const char* keyword, int64_t val, const char* unit = "") {
    std::string buf = keyword;
    buf += "=";
    buf += ToAd(val);
    buf += unit;
    buf += '\n';
    CopyStr(arr, buf);
}

template <size_t N>
void SetAdo06(uc (&arr)[N], const char* keyword, double val, const char* unit = "") {
    std::string buf = keyword;
    buf += "=";
    buf += ToAdo06(val);
    buf += unit;
    buf += '\n';
    CopyStr(arr, buf);
}

template <size_t N>
void SetAdo73(uc (&arr)[N], const char* keyword, double val, const char* unit = "") {
    std::string buf = keyword;
    buf += "=";
    buf += ToAdo73(val);
    buf += unit;
    buf += '\n';
    CopyStr(arr, buf);
}

template <size_t N>
void SetAdo46(uc (&arr)[N], const char* keyword, double val, const char* unit = "") {
    std::string buf = keyword;
    buf += "=";
    buf += ToAdo46(val);
    buf += unit;
    buf += '\n';
    CopyStr(arr, buf);
}

template <size_t N>
void SetAfl(uc (&arr)[N], const char* keyword, float val, const char* unit = "") {
    std::string buf = keyword;
    buf += "=";
    buf += ToAfl(val);
    buf += unit;
    buf += '\n';
    CopyStr(arr, buf);
}

inline std::string PtimeToStr(boost::posix_time::ptime time) {
    boost::posix_time::time_facet* facet = new boost::posix_time::time_facet("%d-%b-%Y %H:%M:%S.%f");
    std::stringstream date_stream;
    date_stream.imbue(std::locale(date_stream.getloc(), facet));
    date_stream << time;
    auto str = date_stream.str();
    for (auto& e : str) e = toupper(e);
    return str;
}

