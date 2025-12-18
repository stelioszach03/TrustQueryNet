#!/usr/bin/env bash
set -euo pipefail

if ! command -v isic >/dev/null 2>&1; then
  echo "The 'isic' CLI is not installed."
  echo "Install it from the official releases: https://github.com/ImageMarkup/isic-cli/releases"
  exit 1
fi

mkdir -p data/ham10000

echo "Listing collections so you can identify the HAM10000 collection id..."
isic collection list

echo
echo "Once you know the collection id, run:"
echo "  isic metadata download --collections <COLLECTION_ID> data/ham10000/"
echo "  isic image download --collections <COLLECTION_ID> data/ham10000/images/"
