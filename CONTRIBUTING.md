# Contributing to YAAP

Thanks for considering a contribution — bug reports, fixes, and features are
all welcome.

## License of contributions

YAAP is licensed under [MIT](LICENSE). **By submitting a contribution (pull
request, patch, or otherwise) you agree to license it under the same MIT
terms as the rest of the project**, and you confirm you have the right to do
so (i.e. it's your own original work, or you have permission to submit it).

No separate CLA or paperwork is required — this is the same "inbound =
outbound" convention used by most MIT-licensed projects, and it keeps the
project's licensing simple and unambiguous for everyone, including future
releases.

## How to contribute

1. Open an issue first for anything non-trivial, so the approach can be
   discussed before you invest time in it.
2. Fork, branch, make your change.
3. Keep PRs focused — one fix or feature per PR is easier to review.
4. Test your change locally (`docker compose up --build`) before opening
   the PR; there's no CI yet, so manual verification matters.

## Reporting bugs

Include: what you did, what you expected, what happened instead, and
whether you're on CPU or GPU (`docker-compose.yml` target). Logs from
`docker compose logs` are usually the fastest way to get a fix.
